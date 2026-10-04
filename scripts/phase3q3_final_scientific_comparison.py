"""Phase 3Q.3 fixed-panel, inference-only S2 versus S1+S2 comparison."""
from __future__ import annotations
import hashlib, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.croma_adapter import CROMAAdapter
from src.dataset_loader import discover_samples,load_sample
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.optical_sar_joint import EXPECTED_JOINT_FINGERPRINT,EXPECTED_JOINT_SHA,load_verified_joint_projector,module_fingerprint
from src.eo_vlm.phase3q3_metrics import CONDITIONS,changed,evidence_gate,outcome,paired_wins,qa_summary,task_classification
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch,freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT,hash_state
SOURCE=ROOT/'artifacts/training/phase3q/phase3q1_fusion_adaptation';OUT=ROOT/'artifacts/training/phase3q/phase3q3_final_scientific_comparison'
MANIFEST=SOURCE/'phase3q1_training_manifest.json';SHUFFLE=SOURCE/'phase3q1_shuffle_map.json';JOINT=SOURCE/'phase3q1_joint_projector_final.pt';S2CK=ROOT/'artifacts/training/phase3o/phase3o12_run/phase3o12_projector_final.pt';S1CK=ROOT/'artifacts/training/phase3p/phase3p1_sar_adaptation/phase3p1_sar_projector_final.pt'
DATA=Path(r'D:\Satquery_ai datasets\comparison\raw-1000');CS=Path(r'D:\Satquery_ai datasets\croma_official');CK=Path(r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt');TASKS=('binary_qa','multiple_choice_qa','caption')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def q(r):return 'Describe this paired Sentinel-1 and Sentinel-2 scene.' if r['task_type']=='caption' else r['question']
def a(r):return r['caption'] if r['task_type']=='caption' else r['answer']
def parse(task,text):
 for x in text.lower().replace('.',' ').replace('(',' ').replace(')',' ').split():
  if task=='binary_qa' and x in {'yes','no'}:return x
  if task=='multiple_choice_qa' and x in {'a','b','c','d'}:return x
 return None
def decode(model,tok,visual,question):
 with torch.inference_mode():
  b=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=visual,questions=[question],answers=None)
  z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=8,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True)
 return tok.decode(z[0,b['input_ids'].shape[1]:],skip_special_tokens=True)
def score(model,tok,visual,question,candidate):
 with torch.inference_mode():
  b=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=visual,questions=[question],answers=[candidate]); loss=float(model(**b).loss)
 return -loss
def candidates(r):return ['yes','no'] if r['task_type']=='binary_qa' else ['a','b','c','d'] if r['task_type']=='multiple_choice_qa' else []
def response(model,tok,visual,r,condition):
 text=decode(model,tok,visual,q(r)); record={'record_id':r['record_id'],'task_type':r['task_type'],'condition':condition,'generated_text':text,'parsed_answer':parse(r['task_type'],text),'target':a(r),'empty':not bool(text.strip()),'output_length':len(text),'visual_token_shape':list(visual.shape),'finite':bool(torch.isfinite(visual).all())}
 cs=candidates(r)
 if cs:
  scores={c:score(model,tok,visual,q(r),c) for c in cs}; target=a(r); record['target_sequence_score']=scores[target]; record['candidate_scores']=scores; record['target_rank']=1+sum(v>scores[target] for k,v in scores.items() if k!=target); record['target_vs_best_distractor_margin']=scores[target]-max(v for k,v in scores.items() if k!=target)
 return record
def main():
 OUT.mkdir(parents=True,exist_ok=True);print('PHASE3Q3 START\nDATASET = BIGEARTHNET ONLY',flush=True)
 mb,sb=sha(MANIFEST),sha(SHUFFLE);man=json.loads(MANIFEST.read_text());sm=json.loads(SHUFFLE.read_text());rows=man['selection']['validation'];maps=sm['maps']
 if len(rows)!=30 or Counter(r['task_type'] for r in rows)!=Counter({t:10 for t in TASKS}) or man['test_access_count']!=0 or sm['test_access_count']!=0:raise RuntimeError('fixed panel integrity failure')
 if any(maps.get(r['record_id'])!=r['shuffled_image_id'] for r in rows):raise RuntimeError('shuffle map mismatch')
 gate={'panel_records':[r['record_id'] for r in rows],'conditions':list(CONDITIONS),'test_access_count':0,'rule':'fixed deterministic gate implemented in src.eo_vlm.phase3q3_metrics.evidence_gate','pre_registered_before_aggregate':True,'automatic_controller_routing':'prohibited'};dump('phase3q3_preregistered_gate.json',gate);print('PANEL = 30\nTEST ACCESS = 0',flush=True)
 joint,jsha,jfp=load_verified_joint_projector(JOINT);assert jsha==EXPECTED_JOINT_SHA and jfp==EXPECTED_JOINT_FINGERPRINT
 samples={x.patch_id:x for x in discover_samples(DATA,strict=True)};prepared={r['image_id']:load_sample(samples[r['image_id']]) for r in rows}
 c=CROMAAdapter(CS,CK,device='cuda');[p.requires_grad_(False) for p in c.model.parameters()];c0=module_fingerprint(c.model)
 runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;freeze_qwen(model);q0=hash_state(model)
 s2=S2MultispectralProjector().cuda().eval();s2.load_state_dict(torch.load(S2CK,map_location='cpu',weights_only=True)['state_dict']);[p.requires_grad_(False) for p in s2.parameters()];s20=module_fingerprint(s2)
 s1=S1SARProjector().eval();s1.load_state_dict(torch.load(S1CK,map_location='cpu',weights_only=True)['state_dict']);[p.requires_grad_(False) for p in s1.parameters()];s10=module_fingerprint(s1);joint0=module_fingerprint(joint)
 records=[]
 for i,r in enumerate(rows,1):
  own=prepared[r['image_id']];other=prepared[maps[r['record_id']]]; pairs={'S1_PLUS_S2_CORRECT':(own.raw_sar,own.raw_optical),'SHUFFLED_S1_CORRECT_S2':(other.raw_sar,own.raw_optical),'CORRECT_S1_SHUFFLED_S2':(own.raw_sar,other.raw_optical),'FULLY_SHUFFLED_S1_S2':(other.raw_sar,other.raw_optical),'ZERO_S1_CORRECT_S2':(np.zeros_like(own.raw_sar),own.raw_optical),'CORRECT_S1_ZERO_S2':(own.raw_sar,np.zeros_like(own.raw_optical))}
  outcomes={}
  visual=s2(torch.from_numpy(own.optical).unsqueeze(0).cuda());outcomes['S2_ONLY_CORRECT']=response(model,tok,visual,r,'S2_ONLY_CORRECT')
  for cond,(sar,optical) in pairs.items():
   with torch.inference_mode():visual=joint(c.infer(optical,sar)['joint_encodings'])
   outcomes[cond]=response(model,tok,visual,r,cond)
  records.append({'record_id':r['record_id'],'image_id':r['image_id'],'shuffled_image_id':maps[r['record_id']],'task_type':r['task_type'],'conditions':outcomes});print(f'[3Q3] record {i}/30',flush=True)
 # Per-task summaries are strict behavior aggregates, never subjective caption quality estimates.
 flat=lambda condition,task:[x['conditions'][condition] for x in records if x['task_type']==task]
 binary={'s2_only':qa_summary(flat('S2_ONLY_CORRECT','binary_qa')),'joint':qa_summary(flat('S1_PLUS_S2_CORRECT','binary_qa')),'paired':paired_wins(flat('S2_ONLY_CORRECT','binary_qa'),flat('S1_PLUS_S2_CORRECT','binary_qa'))};binary['classification']=task_classification(binary['s2_only'],binary['joint']);dump('phase3q3_binary_comparison.json',binary)
 mcq={'s2_only':qa_summary(flat('S2_ONLY_CORRECT','multiple_choice_qa')),'joint':qa_summary(flat('S1_PLUS_S2_CORRECT','multiple_choice_qa')),'paired':paired_wins(flat('S2_ONLY_CORRECT','multiple_choice_qa'),flat('S1_PLUS_S2_CORRECT','multiple_choice_qa'))};mcq['classification']=task_classification(mcq['s2_only'],mcq['joint']);dump('phase3q3_mcq_comparison.json',mcq)
 cap={};
 for condition in CONDITIONS:
  vals=flat(condition,'caption');cap[condition]={'count':len(vals),'empty':sum(x['empty'] for x in vals),'nonempty':sum(not x['empty'] for x in vals),'unique_output_count':len({x['generated_text'] for x in vals}),'mean_output_length':sum(x['output_length'] for x in vals)/len(vals)}
 cap['correct_vs_shuffled_s1_changes']=changed(flat('S1_PLUS_S2_CORRECT','caption'),flat('SHUFFLED_S1_CORRECT_S2','caption'));cap['correct_vs_shuffled_s2_changes']=changed(flat('S1_PLUS_S2_CORRECT','caption'),flat('CORRECT_S1_SHUFFLED_S2','caption'));cap['correct_vs_zero_s1_changes']=changed(flat('S1_PLUS_S2_CORRECT','caption'),flat('ZERO_S1_CORRECT_S2','caption'));cap['correct_vs_zero_s2_changes']=changed(flat('S1_PLUS_S2_CORRECT','caption'),flat('CORRECT_S1_ZERO_S2','caption'));dump('phase3q3_caption_comparison.json',cap)
 changes=lambda task,condition:changed(flat('S1_PLUS_S2_CORRECT',task),flat(condition,task))
 s1c={'vs_shuffled_s1':{t:changes(t,'SHUFFLED_S1_CORRECT_S2') for t in TASKS},'vs_zero_s1':{t:changes(t,'ZERO_S1_CORRECT_S2') for t in TASKS}};s1c['total_behavior_changes']=sum(sum(x.values()) for x in s1c.values());dump('phase3q3_s1_contribution.json',s1c)
 s2c={'vs_shuffled_s2':{t:changes(t,'CORRECT_S1_SHUFFLED_S2') for t in TASKS},'vs_zero_s2':{t:changes(t,'CORRECT_S1_ZERO_S2') for t in TASKS}};s2c['total_behavior_changes']=sum(sum(x.values()) for x in s2c.values());dump('phase3q3_s2_contribution.json',s2c)
 full={t:changes(t,'FULLY_SHUFFLED_S1_S2') for t in TASKS};full['total_behavior_changes']=sum(full.values())
 task_classes={'binary':binary['classification'],'mcq':mcq['classification'],'caption':'NO_CLEAR_DIFFERENCE'};classification,routing=evidence_gate(task_classes=task_classes,s1_changes=s1c['total_behavior_changes'],s2_changes=s2c['total_behavior_changes'],total_records=30)
 immutable={'qwen':hash_state(model)==q0,'croma':module_fingerprint(c.model)==c0,'s2_projector':module_fingerprint(s2)==s20,'s1_projector':module_fingerprint(s1)==s10,'joint_projector':module_fingerprint(joint)==joint0,'manifest':sha(MANIFEST)==mb,'shuffle_map':sha(SHUFFLE)==sb};summary={'status':'PHASE3Q3_COMPLETE','dataset':'BigEarthNet only','panel_count':30,'task_counts':dict(Counter(r['task_type'] for r in rows)),'test_access_count':0,'joint_checkpoint':{'sha256':jsha,'fingerprint':jfp},'task_classifications':task_classes,'scientific_classification':classification,'routing_recommendation':routing,'full_shuffle_behavior_changes':full,'immutability':immutable,'scientific_conclusion':'Fixed-panel behavior only; no claimed benchmark gain or automatic route enablement.'};dump('phase3q3_per_record_comparison.json',{'records':records,'test_access_count':0});dump('phase3q3_summary.json',summary)
 print('BINARY:\nS2 ONLY = '+str(binary['s2_only'])+'\nS1+S2 = '+str(binary['joint'])+'\nJOINT WINS/LOSSES/TIES = '+str(binary['paired']),flush=True);print('MCQ:\nS2 ONLY = '+str(mcq['s2_only'])+'\nS1+S2 = '+str(mcq['joint'])+'\nJOINT WINS/LOSSES/TIES = '+str(mcq['paired']),flush=True);print('CAPTION:\nS2 UNIQUE = '+str(cap['S2_ONLY_CORRECT']['unique_output_count'])+'\nJOINT UNIQUE = '+str(cap['S1_PLUS_S2_CORRECT']['unique_output_count'])+'\nS1 SHUFFLE CHANGES = '+str(cap['correct_vs_shuffled_s1_changes'])+'\nS2 SHUFFLE CHANGES = '+str(cap['correct_vs_shuffled_s2_changes']),flush=True);print('S1 CONTRIBUTION:\nBinary changes = '+str(s1c['vs_shuffled_s1']['binary_qa'])+'\nMCQ changes = '+str(s1c['vs_shuffled_s1']['multiple_choice_qa'])+'\nCaption changes = '+str(s1c['vs_shuffled_s1']['caption']),flush=True);print('S2 CONTRIBUTION:\nBinary changes = '+str(s2c['vs_shuffled_s2']['binary_qa'])+'\nMCQ changes = '+str(s2c['vs_shuffled_s2']['multiple_choice_qa'])+'\nCaption changes = '+str(s2c['vs_shuffled_s2']['caption']),flush=True);print('FULL SHUFFLE EFFECT = '+str(full),flush=True);print('QWEN UNCHANGED = '+('YES' if immutable['qwen'] else 'NO')+'\nCROMA UNCHANGED = '+('YES' if immutable['croma'] else 'NO')+'\nPROJECTORS UNCHANGED = '+('YES' if all(immutable[k] for k in ('s1_projector','s2_projector','joint_projector')) else 'NO')+'\nTEST ACCESS = 0',flush=True);print('SCIENTIFIC CLASSIFICATION = '+classification+'\nROUTING RECOMMENDATION = '+routing+'\nPHASE3Q3 COMPLETE',flush=True)
if __name__=='__main__':main()
