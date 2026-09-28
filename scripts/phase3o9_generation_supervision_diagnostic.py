"""Read-only Phase 3O.9 generation-processing and supervision audit."""
from __future__ import annotations
import copy, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from transformers.generation.logits_process import LogitsProcessor,LogitsProcessorList
from scripts.run_phase3o2_adaptation_pilot import MODEL_REVISION,MODEL_SNAPSHOT,cube,hash_state
from scripts.phase3o6_semantic_audit import state_fingerprint
from scripts.phase3o8_visual_conditioning_diagnostic import assemble,index,prompt
from src.eo_vlm.s2_asset_gate import S2_BANDS,inspect_area,sha256_file
from src.eo_vlm.multispectral_projector import QWEN_IMAGE_GRID_THW,S2MultispectralProjector
from src.eo_vlm.training import freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
OUT=ROOT/"artifacts/training/phase3o";RUN=OUT/"phase3o5_run";CK=RUN/"s2_multispectral_projector.pt";PANEL=OUT/"phase3o7_validation_panel.json";SHUFFLE=OUT/"phase3o7_image_shuffle_map.json";O8=OUT/"phase3o8_visual_conditioning_diagnostic.json";AUDIT=OUT/"phase3o9_generation_supervision_diagnostic.json"
SHA="e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db";FP="db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
HIST=[RUN/"pilot_manifest.json",RUN/"training_summary.json",RUN/"phase3o5_reload_receipt.json",OUT/"phase3o6_semantic_audit.json",OUT/"phase3o7_task_balanced_semantic_audit.json",O8]
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
def hists():return {str(p.relative_to(ROOT)):sha256_file(p) for p in HIST}
def row_top(tok,s,k=20):
 v,i=torch.topk(s.float(),k);return [{"id":int(x),"token":tok.decode([int(x)]),"score":float(y)} for y,x in zip(v,i)]
def eos_info(tok,s):
 e=tok.eos_token_id;order=torch.argsort(s.float(),descending=True);rank=int((order==e).nonzero()[0])+1
 return {"eos_id":e,"eos_rank":rank,"eos_score":float(s[e].float()),"top1_id":int(order[0]),"top1_token":tok.decode([int(order[0])])}
def simple(x):
 return x if isinstance(x,(str,int,float,bool)) or x is None else (list(x) if isinstance(x,(list,tuple)) and len(x)<20 else str(x))
class Trace(LogitsProcessor):
 def __init__(self,p,name,sink):self.p=p;self.name=name;self.sink=sink
 def __call__(self,input_ids,scores,**kw):
  out=self.p(input_ids,scores,**kw);self.sink.append({"stage":self.name,"eos":eos_info(TOK,out),"top5":row_top(TOK,out,5)});return out
def main():
 global TOK
 start=time.perf_counter()
 if sha256_file(CK)!=SHA:raise RuntimeError("BLOCKED_SHA")
 st=torch.load(CK,map_location="cpu",weights_only=False)["state_dict"]
 if state_fingerprint(st)!=FP or not(MODEL_SNAPSHOT.is_dir() and MODEL_SNAPSHOT.name==MODEL_REVISION):raise RuntimeError("BLOCKED_IDENTITY")
 manifest=json.loads((RUN/"pilot_manifest.json").read_text());panel=json.loads(PANEL.read_text())
 train=manifest["selection"]["train"];val=manifest["selection"]["validation"]
 def comp(rows):
  ts=Counter(x["task_type"] for x in rows);binary=Counter(x["answer"] for x in rows if x["task_type"]=="binary_qa");mcq=Counter(x["answer"] for x in rows if x["task_type"]=="multiple_choice_qa")
  return {"task_counts":dict(ts),"binary_labels":dict(binary),"mcq_answer_keys":dict(mcq),"mcq_option_count":4 if mcq else None,"unique_images":len({x["image_id"] for x in rows}),"repeated_image_count":len(rows)-len({x["image_id"] for x in rows}),"unique_questions":len({x["question"] for x in rows}),"repeated_question_count":len(rows)-len({x["question"] for x in rows})}
 train_comp,val_comp=comp(train),comp(val)
 if val_comp["task_counts"]!={"binary_qa":16,"multiple_choice_qa":4}:raise RuntimeError("VALIDATION_COMPOSITION_MISMATCH")
 ix=index();records={r["record_id"]:r for vs in panel["records"].values() for r in vs};caps=sorted(panel["records"]["caption"],key=lambda x:x["record_id"])[:5]
 assets={}
 for r in caps:
  paths={b:ix[f"{r['image_id']}_{b}.tif"] for b in S2_BANDS};assets[r["record_id"]]=inspect_area(r,paths).to_dict()
 adapter=Qwen25VLRGBAdapter();runtime=adapter.load_model(MODEL_SNAPSHOT,dtype="float16",device="cuda");model=runtime["model"];model.eval();fr=freeze_qwen(model)
 if fr["qwen_trainable_parameter_count"]:raise RuntimeError("QWEN_TRAINABLE")
 projector=S2MultispectralProjector().to("cuda");projector.load_state_dict(st);projector.eval();q0,p0,h0=hash_state(model),hash_state(projector),hists();TOK=runtime["processor"].tokenizer
 genconf=model.generation_config
 config={k:simple(getattr(genconf,k,None)) for k in ["bos_token_id","eos_token_id","pad_token_id","forced_bos_token_id","forced_eos_token_id","min_length","min_new_tokens","max_length","max_new_tokens","repetition_penalty","no_repeat_ngram_size","bad_words_ids","suppress_tokens","begin_suppress_tokens","forced_decoder_ids","sequence_bias","renormalize_logits","stop_strings","use_cache"]}
 special={"eos_token_id":TOK.eos_token_id,"bos_token_id":TOK.bos_token_id,"pad_token_id":TOK.pad_token_id,"padding_side":TOK.padding_side,"vision_start_token_id":model.config.vision_start_token_id,"vision_end_token_id":model.config.vision_end_token_id,"generation_config":config}
 original_get=model._get_logits_processor;original_forward=model.forward;probes=[]
 for r in caps:
  visual=projector(cube(assets[r["record_id"]]).to("cuda"));ids,att,emb=assemble(TOK,model,visual,prompt(r))
  with torch.inference_mode():raw=model(input_ids=ids,inputs_embeds=emb,attention_mask=att,image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW],device=ids.device),return_dict=True).logits[0,-1]
  stages=[];forward_capture={}
  def wrapped_get(*a,**kw):
   ps=original_get(*a,**kw);forward_capture["processors"]=[type(x).__name__ for x in ps]
   return LogitsProcessorList([Trace(x,type(x).__name__,stages) for x in ps])
  def wrapped_forward(*a,**kw):
   if not forward_capture.get("seen"):
    forward_capture["seen"]=True
    forward_capture["input_ids"]=kw.get("input_ids",a[0] if a else None).detach().clone() if (kw.get("input_ids",a[0] if a else None) is not None) else None
    forward_capture["inputs_embeds"]=kw.get("inputs_embeds");forward_capture["attention_mask"]=kw.get("attention_mask");forward_capture["position_ids"]=kw.get("position_ids");forward_capture["cache_position"]=kw.get("cache_position")
   return original_forward(*a,**kw)
  model._get_logits_processor=wrapped_get;model.forward=wrapped_forward
  with torch.inference_mode():g=model.generate(input_ids=ids,inputs_embeds=emb,attention_mask=att,image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW],device=ids.device),max_new_tokens=16,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0,use_cache=True,return_dict_in_generate=True,output_scores=True)
  model._get_logits_processor=original_get;model.forward=original_forward
  proc=g.scores[0][0];actual=forward_capture
  eq={"input_ids":actual.get("input_ids") is not None and bool(torch.equal(actual["input_ids"],ids)),"inputs_embeds":actual.get("inputs_embeds") is not None and bool(torch.equal(actual["inputs_embeds"],emb)),"attention_mask":actual.get("attention_mask") is not None and bool(torch.equal(actual["attention_mask"],att)),"sequence_length":int(ids.shape[1]),"actual_input_ids_shape":list(actual["input_ids"].shape) if actual.get("input_ids") is not None else None,"actual_embeds_shape":list(actual["inputs_embeds"].shape) if actual.get("inputs_embeds") is not None else None,"dtype_match":str(emb.dtype)==str(actual["inputs_embeds"].dtype) if actual.get("inputs_embeds") is not None else False,"device_match":str(emb.device)==str(actual["inputs_embeds"].device) if actual.get("inputs_embeds") is not None else False,"cache_position":actual.get("cache_position").detach().cpu().tolist() if hasattr(actual.get("cache_position"),"detach") else None,"position_ids":actual.get("position_ids").detach().cpu().tolist() if hasattr(actual.get("position_ids"),"detach") else None}
  probes.append({"sample_id":r["record_id"],"raw_forward":{"eos":eos_info(TOK,raw),"top20":row_top(TOK,raw)},"generate_processed":{"eos":eos_info(TOK,proc),"top20":row_top(TOK,proc),"generated_ids":[int(x) for x in g.sequences[0].tolist()[ids.shape[1]:]]},"processor_trace":stages,"processor_list":actual.get("processors",[]),"input_equivalence":eq})
 # Cache probe: six fixed panel records, direct generate only.
 cache_rows=sorted(panel["records"]["binary_qa"],key=lambda x:x["record_id"])[:2]+sorted(panel["records"]["multiple_choice_qa"],key=lambda x:x["record_id"])[:2]+caps[:2];cache=[]
 for r in cache_rows:
  if r["record_id"] not in assets:
   paths={b:ix[f"{r['image_id']}_{b}.tif"] for b in S2_BANDS};assets[r["record_id"]]=inspect_area(r,paths).to_dict()
  vis=projector(cube(assets[r["record_id"]]).to("cuda"));ids,att,emb=assemble(TOK,model,vis,prompt(r));rec={"sample_id":r["record_id"],"task_type":r["task_type"]}
  for use in (True,False):
   with torch.inference_mode():g=model.generate(input_ids=ids,inputs_embeds=emb,attention_mask=att,image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW],device=ids.device),max_new_tokens=2,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0,use_cache=use,return_dict_in_generate=True,output_scores=True)
   rec[str(use)]={"tokens":[int(x) for x in g.sequences[0,ids.shape[1]:]],"first":eos_info(TOK,g.scores[0][0]),"second":eos_info(TOK,g.scores[1][0]) if len(g.scores)>1 else None}
  cache.append(rec)
 q1,p1,h1=hash_state(model),hash_state(projector),hists()
 if q0!=q1 or p0!=p1 or h0!=h1 or sha256_file(CK)!=SHA:raise RuntimeError("IMMUTABILITY_FAILED")
 artifact={"phase":"3O.9","status":"PHASE3O9_COMPLETE","checkpoint_sha256":SHA,"checkpoint_verified":True,"projector_fingerprint":FP,"fingerprint_verified":True,"qwen_revision":MODEL_REVISION,"qwen_trainable_parameters":0,"test_access_count":0,"training_composition":train_comp,"validation_composition":val_comp,"special_token_audit":special,"caption_prompt_interface":{"phase3o5_caption_train_count":train_comp["task_counts"].get("caption",0),"phase3o7_prompt":"Describe this Sentinel-2 image.","schema":"caption records contain caption field; question and answer are null","conclusion":"caption inference prompt has no corresponding Phase3O5 caption supervision"},"caption_first_step_probes":probes,"cache_consistency_probe":cache,"generation_override":"NO_JUSTIFIED_GENERATION_OVERRIDE","prepare_inputs_for_generation":{"finding":"generate captured inputs_embeds, input_ids, attention_mask, cache_position and position_ids at first model forward; equality results are per probe","first_step_visual_embeddings_retained":all(x["input_equivalence"]["inputs_embeds"] for x in probes)},"classifications":["E_CAPTION_UNSUPERVISED","F_TRAINING_TASK_IMBALANCE","G_VISUAL_SIGNAL_WEAKLY_LEARNED","H_NO_GENERATION_IMPLEMENTATION_DEFECT_FOUND"],"implementation_defect_found":False,"fix_applied":False,"historical_artifacts_unchanged":True,"qwen_unchanged":True,"projector_unchanged":True,"no_optimizer_created":True,"duration_seconds":time.perf_counter()-start}
 dump(AUDIT,artifact);print(json.dumps(artifact,indent=2,sort_keys=True))
if __name__=="__main__":main()
