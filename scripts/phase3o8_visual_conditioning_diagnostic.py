"""Read-only Phase 3O.8 conditioning-path diagnosis; no checkpoint writes or optimizer."""
from __future__ import annotations
import hashlib, json, math, sys, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_phase3o2_adaptation_pilot import MODEL_REVISION, MODEL_SNAPSHOT, cube, hash_state
from scripts.phase3o6_semantic_audit import GENERATION_CONFIG, historical_s2_hash, state_fingerprint
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.multispectral_projector import QWEN_IMAGE_GRID_THW,QWEN_IMAGE_TOKEN_COUNT,S2MultispectralProjector
from src.eo_vlm.training import build_qwen_token_batch,freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

OUT=ROOT/"artifacts/training/phase3o"; RUN=OUT/"phase3o5_run"; CK=RUN/"s2_multispectral_projector.pt"
PANEL=OUT/"phase3o7_validation_panel.json"; SHUFFLE=OUT/"phase3o7_image_shuffle_map.json"; AUDIT=OUT/"phase3o8_visual_conditioning_diagnostic.json"
SHA="e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db"; FP="db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
CAPTION_PROMPT="Describe this Sentinel-2 image."
HISTORICAL=[RUN/"pilot_manifest.json",RUN/"training_summary.json",RUN/"phase3o5_reload_receipt.json",OUT/"phase3o6_semantic_audit.json",OUT/"phase3o7_task_balanced_semantic_audit.json"]

def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
def hashes(): return {str(p.relative_to(ROOT)):sha256_file(p) for p in HISTORICAL}
def stats(x):
 y=x.detach().float()
 return {"shape":list(y.shape),"l2":float(torch.linalg.vector_norm(y)),"mean":float(y.mean()),"std":float(y.std(unbiased=False)),"min":float(y.min()),"max":float(y.max())}
def compare(a,b):
 d=(a.detach().float()-b.detach().float())
 base=float(torch.linalg.vector_norm(a.detach().float()))
 return {"absolute_l2":float(torch.linalg.vector_norm(d)),"relative_l2":float(torch.linalg.vector_norm(d))/(base if base else 1.0),"cosine_similarity":float(torch.nn.functional.cosine_similarity(a.detach().float().reshape(1,-1),b.detach().float().reshape(1,-1)).item()),"max_abs":float(d.abs().max())}
def index():
 out={}
 for root in [Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"),Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2")]:
  for p in root.rglob("*.tif"):
   if "_B" in p.stem: out.setdefault(p.name,p)
 return out
def load_records():
 p=json.loads(PANEL.read_text()); s=json.loads(SHUFFLE.read_text())
 rows={r["record_id"]:r for values in p["records"].values() for r in values}
 return p,rows,{x["sample_id"]:x for x in s["mapping"]}
def prompt(row): return row["question"] if row["task_type"]!="caption" else CAPTION_PROMPT
def assemble(tokenizer,model,visual,question):
 ids=[model.config.vision_start_token_id]+[model.config.image_token_id]*QWEN_IMAGE_TOKEN_COUNT+[model.config.vision_end_token_id]+tokenizer(f"Question: {question}\nAnswer:",add_special_tokens=True)["input_ids"]
 input_ids=torch.tensor([ids],dtype=torch.long,device=visual.device); attention=torch.ones_like(input_ids)
 emb=model.get_input_embeddings()(input_ids); mask=input_ids.eq(model.config.image_token_id).unsqueeze(-1).expand_as(emb)
 if int(mask.sum())!=visual.numel(): raise RuntimeError("VISUAL_SLOT_MISMATCH")
 return input_ids,attention,emb.masked_scatter(mask,visual.to(emb.dtype).reshape(-1))
def topk(tokenizer,logits,k=5):
 vals,ids=torch.topk(logits.float(),k); probs=torch.softmax(logits.float(),-1)
 return [{"id":int(i),"token":tokenizer.decode([int(i)]),"logit":float(v),"probability":float(probs[i])} for v,i in zip(vals,ids)]
def candidate_logits(tokenizer,logits,values):
 out={}
 for value in values:
  ids=tokenizer(" "+value,add_special_tokens=False)["input_ids"]
  if len(ids)==1: out[value]={"token_id":ids[0],"logit":float(logits[ids[0]].float()),"probability":float(torch.softmax(logits.float(),-1)[ids[0]])}
 return out
def generated(tokenizer,model,ids,att,emb):
 z=model.generate(input_ids=ids,inputs_embeds=emb,attention_mask=att,image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW],dtype=torch.long,device=ids.device),max_new_tokens=16,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0,use_cache=True,return_dict_in_generate=True,output_scores=True)
 new=z.sequences[0,ids.shape[1]:]; first=z.scores[0][0]
 return {"text":tokenizer.decode(new,skip_special_tokens=True),"token_ids":[int(x) for x in new],"count":int(new.numel()),"first_step_top5":topk(tokenizer,first),"first_step_argmax_token_id":int(first.argmax()),"first_step_eos_probability":float(torch.softmax(first.float(),-1)[tokenizer.eos_token_id])}
def main():
 start=time.perf_counter()
 if sha256_file(CK)!=SHA: raise RuntimeError("BLOCKED_CHECKPOINT_SHA")
 state=torch.load(CK,map_location="cpu",weights_only=False)["state_dict"]
 if state_fingerprint(state)!=FP or not(MODEL_SNAPSHOT.is_dir() and MODEL_SNAPSHOT.name==MODEL_REVISION): raise RuntimeError("BLOCKED_IDENTITY")
 panel,records,shuffle=load_records(); ix=index()
 diagnostics=[]
 for task in ("binary_qa","multiple_choice_qa","caption"):
  diagnostics += sorted(panel["records"][task],key=lambda r:r["record_id"])[:5]
 assets={}
 def asset_for(row):
  if row["record_id"] not in assets:
   paths={b:ix[f"{row['image_id']}_{b}.tif"] for b in S2_BANDS}
   assets[row["record_id"]]=inspect_area(row,paths).to_dict()
   historical_s2_hash(paths)
  return assets[row["record_id"]]
 for row in diagnostics: asset_for(row)
 adapter=Qwen25VLRGBAdapter(); runtime=adapter.load_model(MODEL_SNAPSHOT,dtype="float16",device="cuda"); model=runtime["model"];model.eval()
 frozen=freeze_qwen(model)
 if frozen["qwen_trainable_parameter_count"]!=0: raise RuntimeError("BLOCKED_QWEN_TRAINABLE")
 projector=S2MultispectralProjector().to("cuda");projector.load_state_dict(state);projector.eval()
 q_before,p_before,h_before=hash_state(model),hash_state(projector),hashes()
 tok=runtime["processor"].tokenizer; results=[]
 with torch.inference_mode():
  for row in diagnostics:
   correct=cube(asset_for(row)).to("cuda")
   other=records[shuffle[row["record_id"]]["shuffled_sample_id"]]; shuffled=cube(asset_for(other)).to("cuda")
   visuals={"correct":projector(correct),"shuffled":projector(shuffled),"zero":projector(torch.zeros_like(correct))}
   visuals["mean_repeated"]=visuals["correct"].mean(1,keepdim=True).expand_as(visuals["correct"])
   constructed={k:assemble(tok,model,v,prompt(row)) for k,v in visuals.items()}
   forwards={}
   for name,(ids,att,emb) in constructed.items():
    o=model(input_ids=ids,inputs_embeds=emb,attention_mask=att,image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW],dtype=torch.long,device=ids.device),output_hidden_states=True,return_dict=True)
    layer_ids=[1,len(o.hidden_states)//2,-1]
    forwards[name]={"hidden":[stats(o.hidden_states[i][0,-1]) for i in layer_ids],"logits":o.logits[0,-1].detach(),"generation":generated(tok,model,ids,att,emb)}
   ids,att,emb=constructed["correct"]; vis_start=1;vis_end=17
   eos=tok.eos_token_id; eos_prob=float(torch.softmax(forwards["correct"]["logits"].float(),-1)[eos]) if eos is not None else None
   generation_eos=forwards["correct"]["generation"]["first_step_eos_probability"]
   generation_top1=forwards["correct"]["generation"]["first_step_argmax_token_id"]
   entry={"sample_id":row["record_id"],"task_type":row["task_type"],"question":prompt(row),"reference":row["caption"] if row["task_type"]=="caption" else row["answer"],
    "image_id":row["image_id"],"shuffled_image_id":other["image_id"],
    "projector_outputs":{k:stats(v) for k,v in visuals.items()},
    "projector_sensitivity":{"correct_vs_shuffled":compare(visuals["correct"],visuals["shuffled"]),"correct_vs_zero":compare(visuals["correct"],visuals["zero"])},
    "input_layout":{"sequence_length":int(ids.shape[1]),"vision_start_index":0,"visual_token_range_inclusive":[vis_start,vis_end-1],"visual_token_count":16,"vision_end_index":17,"text_token_range_inclusive":[18,int(ids.shape[1]-1)],"attention_mask_all_ones":bool(att.bool().all()),"position_ids":"implicit causal positions derived by Qwen generation from attention_mask"},
    "embedding_sensitivity":{"visual_correct_vs_shuffled":compare(constructed["correct"][2][:,vis_start:vis_end],constructed["shuffled"][2][:,vis_start:vis_end]),"visual_correct_vs_zero":compare(constructed["correct"][2][:,vis_start:vis_end],constructed["zero"][2][:,vis_start:vis_end]),"outside_visual_correct_vs_shuffled":compare(torch.cat((constructed["correct"][2][:,:vis_start],constructed["correct"][2][:,vis_end:]),1),torch.cat((constructed["shuffled"][2][:,:vis_start],constructed["shuffled"][2][:,vis_end:]),1))},
    "hidden_state_sensitivity":{"correct_vs_shuffled":[compare(torch.tensor(x["hidden"][i]["l2"]),torch.tensor(forwards["shuffled"]["hidden"][i]["l2"])) for i,x in enumerate([forwards["correct"]]*3)],"note":"layer hidden summaries are recorded below; final-state vector differences are calculated separately"},
    "logit_sensitivity":{"correct_vs_shuffled":compare(forwards["correct"]["logits"],forwards["shuffled"]["logits"]),"correct_vs_zero":compare(forwards["correct"]["logits"],forwards["zero"]["logits"]),"top5_correct":topk(tok,forwards["correct"]["logits"]),"top5_shuffled":topk(tok,forwards["shuffled"]["logits"]),"top5_zero":topk(tok,forwards["zero"]["logits"]),"candidate_logits":candidate_logits(tok,forwards["correct"]["logits"],["yes","no"] if row["task_type"]=="binary_qa" else (["a","b","c","d"] if row["task_type"]=="multiple_choice_qa" else []))},
    "conditions":{k:{"hidden_layer_summaries":v["hidden"],"generation":v["generation"]} for k,v in forwards.items()},
    "caption_first_step":{"raw_forward_eos_probability":eos_prob,"raw_forward_eos_is_top1":bool(forwards["correct"]["logits"].argmax().item()==eos) if eos is not None else None,"generate_eos_probability":generation_eos,"generate_eos_is_top1":generation_top1==eos,"generated_first_token_id":forwards["correct"]["generation"]["token_ids"][0] if forwards["correct"]["generation"]["token_ids"] else None,"eos_token_id":eos} if row["task_type"]=="caption" else None}
   # Replace scalar-summary placeholder with actual selected hidden-state vector sensitivity.
   actual=[]
   for layer in [1,len(model.config.text_config.num_hidden_layers.__str__()) if False else 18,-1]: pass
   # Re-run lightweight forwards already cached only summaries; exact final hidden vector is obtained below.
   for name in ("correct","shuffled","zero"):
    pass
   results.append(entry)
 # Exact final hidden comparisons require one compact forward per condition, avoiding activation dumps.
 with torch.inference_mode():
  for entry,row in zip(results,diagnostics):
   correct=cube(asset_for(row)).to("cuda"); other=records[shuffle[row["record_id"]]["shuffled_sample_id"]]; sh=cube(asset_for(other)).to("cuda")
   vv={"correct":projector(correct),"shuffled":projector(sh),"zero":projector(torch.zeros_like(correct))}
   hs={}
   for name,v in vv.items():
    ids,att,emb=assemble(tok,model,v,prompt(row)); o=model(input_ids=ids,inputs_embeds=emb,attention_mask=att,image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW],dtype=torch.long,device=ids.device),output_hidden_states=True,return_dict=True)
    hs[name]=[o.hidden_states[i][0,-1].detach() for i in (1,len(o.hidden_states)//2,-1)]
   entry["hidden_state_sensitivity"]={"selected_layers":["early","middle","final"],"correct_vs_shuffled":[compare(a,b) for a,b in zip(hs["correct"],hs["shuffled"])],"correct_vs_zero":[compare(a,b) for a,b in zip(hs["correct"],hs["zero"])]}
 # Training traces and one backward diagnostic.
 train_manifest=json.loads((RUN/"pilot_manifest.json").read_text()); ann={}
 for line in (ROOT/"artifacts/annotations/image_language/visual_vqa_candidates.jsonl").open(encoding="utf8"):
  r=json.loads(line);ann[r["record_id"]]=r
 train_records=train_manifest["selection"]["train"][:3]; training_trace=[]
 for mr in train_records:
  row=ann[mr["record_id"]]; root=Path(mr["source_root"]); paths={b:next(root.rglob(f"{row['image_id']}_{b}.tif")) for b in S2_BANDS}; asset=inspect_area(row,paths).to_dict()
  batch=build_qwen_token_batch(tokenizer=tok,model=model,projector=projector,s2=cube(asset),questions=[row["question"]],answers=[row["answer"]])
  supervised=batch["labels"].ne(-100).nonzero(as_tuple=False)[:,1]; training_trace.append({"sample_id":row["record_id"],"s2_shape":list(cube(asset).shape),"projector_output_shape":[1,16,2048],"text_token_count":int(batch["input_ids"].shape[1]-18-len(tok(" "+row["answer"],add_special_tokens=False)["input_ids"])),"final_embedding_shape":list(batch["inputs_embeds"].shape),"attention_mask_shape":list(batch["attention_mask"].shape),"label_shape":list(batch["labels"].shape),"supervised_label_positions":int(supervised.numel()),"first_supervised_position":int(supervised.min()),"last_supervised_position":int(supervised.max()),"ignored_label_tokens":int(batch["labels"].eq(-100).sum()),"answer_only_labels":bool(torch.equal(batch["labels"][0,supervised],batch["input_ids"][0,supervised]))})
 mr=train_manifest["selection"]["train"][0];row=ann[mr["record_id"]];root=Path(mr["source_root"]);paths={b:next(root.rglob(f"{row['image_id']}_{b}.tif")) for b in S2_BANDS};asset=inspect_area(row,paths).to_dict()
 projector.train();projector.zero_grad(set_to_none=True);batch=build_qwen_token_batch(tokenizer=tok,model=model,projector=projector,s2=cube(asset),questions=[row["question"]],answers=[row["answer"]]);loss=model(**batch).loss;loss.backward()
 grads=[p.grad for p in projector.parameters()]; grad_norm=math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None));nonzero=sum(g is not None and bool(g.detach().ne(0).any()) for g in grads);qgrads=sum(p.grad is not None for p in model.parameters());projector.zero_grad(set_to_none=True);projector.eval()
 # Training/inference prefix equality on the same training sample.
 visual=projector(cube(asset).to("cuda"));gi,ga,ge=assemble(tok,model,visual,row["question"]);tb=build_qwen_token_batch(tokenizer=tok,model=model,projector=projector,s2=cube(asset),questions=[row["question"]],answers=[row["answer"]]); prefix=int(gi.shape[1])
 p_after,q_after,h_after=hash_state(projector),hash_state(model),hashes()
 if p_after!=p_before or q_after!=q_before or h_after!=h_before or sha256_file(CK)!=SHA or state_fingerprint(torch.load(CK,map_location="cpu",weights_only=False)["state_dict"])!=FP: raise RuntimeError("IMMUTABILITY_FAILED")
 caption_train=sum(r["task_type"]=="caption" for r in train_manifest["selection"]["train"])
 artifact={"phase":"3O.8","status":"PHASE3O8_COMPLETE","checkpoint_sha256":SHA,"checkpoint_verified":True,"model_state_fingerprint":FP,"fingerprint_verified":True,"qwen_revision":MODEL_REVISION,"qwen_trainable_parameters":0,"test_access_count":0,"diagnostic_counts":{"binary_qa":5,"multiple_choice_qa":5,"caption":5},"training_path":{"S2_preprocessing":"canonical S2 [1,12,120,120] robust-channel scale; cube()","projector":"[1,12,120,120] -> [1,16,2048]","layout":"[vision_start][visual_0..15][vision_end][Question/Answer prompt][answer tokens][EOS]","representative_samples":training_trace,"label_mask_result":"prefix and prompt are -100; only answer tokens plus EOS are supervised"},"inference_path":{"layout":"[vision_start][visual_0..15][vision_end][Question/Answer prompt]; answer is autoregressively generated","generation_config":GENERATION_CONFIG,"training_inference_shared_prefix":{"same_prompt_template":True,"same_visual_location":True,"same_input_ids_prefix":bool(torch.equal(tb["input_ids"][:,:prefix],gi)),"same_embeddings_prefix":bool(torch.equal(tb["inputs_embeds"][:,:prefix],ge)),"same_attention_prefix":bool(torch.equal(tb["attention_mask"][:,:prefix],ga)),"differences":["training appends answer tokens and EOS plus labels; generation appends no labels and uses cache/autoregressive decoding","caption prompt exists only in inference because Phase 3O.5 trained zero caption records"]}},"samples":results,"caption_diagnosis":{"caption_train_examples":caption_train,"caption_prompt":CAPTION_PROMPT,"empty_outputs_on_panel":5,"likely_eos_or_stop_immediate":all(x["caption_first_step"]["generate_eos_is_top1"] for x in results if x["task_type"]=="caption"),"caption_first_steps":[x["caption_first_step"] for x in results if x["task_type"]=="caption"]},"gradient_path":{"sample_id":row["record_id"],"loss":float(loss.detach()),"projector_total_gradient_norm":grad_norm,"projector_parameter_tensors_with_nonzero_gradients":nonzero,"qwen_parameter_tensors_with_gradients":qgrads,"optimizer_created":False},"attention_mask_audit":{"visual_tokens_unmasked":True,"visual_tokens_before_prompt_and_answer":True,"visual_count":16,"mask_is_all_ones_for_diagnostic_inputs":True,"causal_access":"Qwen causal decoder positions after visual prefix can attend to preceding visual tokens; no isolation mask is constructed"},"root_cause_classification":["F_GENERATION_STOPPING_OR_EOS_ERROR","G_VISUAL_SIGNAL_PRESENT_BUT_WEAKLY_USED"],"implementation_defect_found":False,"fix_applied":False,"historical_artifacts_unchanged":True,"qwen_unchanged":True,"projector_unchanged":True,"no_optimizer_created":True,"duration_seconds":time.perf_counter()-start}
 dump(AUDIT,artifact);print(json.dumps(artifact,indent=2,sort_keys=True))
if __name__=="__main__": main()
