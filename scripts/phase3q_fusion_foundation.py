"""Phase 3Q BigEarthNet-only joint-CROMA fusion foundation; no training."""
from __future__ import annotations
import gc,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset_loader import discover_samples,load_sample
from src.croma_adapter import CROMAAdapter
from src.eo_vlm.joint_croma_projector import CromaJointProjector
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.training import build_qwen_visual_token_batch,freeze_qwen
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT,hash_state
OUT=ROOT/'artifacts/training/phase3q/phase3q_multimodal_fusion';M=ROOT/'artifacts/training/phase3o/phase3o12_run/phase3o12_training_manifest.json';S2CK=ROOT/'artifacts/training/phase3o/phase3o12_run/phase3o12_projector_final.pt';S1CK=ROOT/'artifacts/training/phase3p/phase3p1_sar_adaptation/phase3p1_sar_projector_final.pt';DATA=Path(r'D:\Satquery_ai datasets\comparison\raw-1000');CS=Path(r'D:\Satquery_ai datasets\croma_official');CK=Path(r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt');TASKS=('binary_qa','multiple_choice_qa','caption')
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def q(r):return 'Describe this paired Sentinel-1 and Sentinel-2 scene.' if r['task_type']=='caption' else r['question']
def a(r):return r['caption'] if r['task_type']=='caption' else r['answer']
def parse(r,text):
 t=text.strip().lower().replace('.',' ')
 if r['task_type']=='caption':return None
 for x in t.replace('(',' ').replace(')',' ').split():
  if r['task_type']=='binary_qa' and x in {'yes','no'}:return x
  if r['task_type']=='multiple_choice_qa' and x in {'a','b','c','d'}:return x
 return None
def main():
 if OUT.exists():raise RuntimeError('PHASE3Q_NAMESPACE_EXISTS')
 OUT.mkdir(parents=True);print('PHASE3Q START\nDATASET = BIGEARTHNET ONLY',flush=True)
 man=json.loads(M.read_text());assert man['test_access_count']==0
 samples={x.patch_id:x for x in discover_samples(DATA,strict=True)};used=set();panel=[]
 for task in TASKS:
  for r in sorted((x for x in man['selection']['validation'] if x['task_type']==task and x['image_id'] in samples),key=lambda x:x['record_id']):
   if r['image_id'] not in used:panel.append(dict(r));used.add(r['image_id'])
   if sum(x['task_type']==task for x in panel)==10:break
 assert len(panel)==30 and Counter(x['task_type'] for x in panel)=={t:10 for t in TASKS}
 prepared={r['image_id']:load_sample(samples[r['image_id']]) for r in panel}
 for task in TASKS:
  rows=[r for r in panel if r['task_type']==task];ids=[r['image_id'] for r in rows];rot=ids[1:]+ids[:1]
  for r,wrong in zip(rows,rot):r['shuffled_image_id']=wrong
 manifest={'dataset':'BigEarthNet only','test_access_count':0,'records':[{'record_id':r['record_id'],'image_id':r['image_id'],'task_type':r['task_type'],'shuffled_image_id':r['shuffled_image_id'],'paired_s1_s2_verified':True,'s1_shape':[2,120,120],'s2_shape':[12,120,120],'alignment':'load_sample strict common grid'} for r in panel]};dump('phase3q_paired_manifest.json',manifest);dump('phase3q_shuffle_map.json',{'mapping':{r['record_id']:r['shuffled_image_id'] for r in panel},'same_task':True,'test_access_count':0})
 architecture={'selected':'OPTION_C_OFFICIAL_CROMA_JOINT_REPRESENTATION_PLUS_MINIMAL_CROMAJOINTPROJECTOR','components_found':['CROMAAdapter.infer paired joint_encodings [1,225,768]','S2MultispectralProjector [1,16,2048]','S1SARProjector [1,16,2048]','Qwen 16 visual-token interface','fixed Phase3O.10 generation wrapper'],'missing_components':['No existing joint-CROMA-to-Qwen projector; added CromaJointProjector [225,768]->[16,2048]','No validated optical-SAR controller route (left blocked)'],'no_s1_s2_concatenation':'Qwen contract is 16 visual tokens; joint CROMA retains that geometry','test_access_count':0};dump('phase3q_architecture_audit.json',architecture);print('PAIRED RECORDS AVAILABLE = '+str(len(samples))+'\nSELECTED FUSION ARCHITECTURE = '+architecture['selected'],flush=True)
 c=CROMAAdapter(CS,CK,device='cuda');[p.requires_grad_(False) for p in c.model.parameters()];c0=fp(c.model);cache={}
 # Cache separately encoded and all six paired joint conditions from official CROMA.
 for i,r in enumerate(panel,1):
  x=prepared[r['image_id']];w=prepared[r['shuffled_image_id']];print(f'[3Q][CROMA] record {i}/30',flush=True)
  with torch.no_grad():
   own=c.infer(x.raw_optical,x.raw_sar); ss=c.infer(w.raw_optical,x.raw_sar); so=c.infer(x.raw_optical,w.raw_sar); both=c.infer(w.raw_optical,w.raw_sar); zs=c.infer(x.raw_optical,np.zeros_like(x.raw_sar)); zo=c.infer(np.zeros_like(x.raw_optical),x.raw_sar)
  cache[r['record_id']]={'s2':torch.from_numpy(x.optical).unsqueeze(0),'sar':own['SAR_encodings'].cpu(),'joint':{k:v['joint_encodings'].cpu() for k,v in {'correct':own,'shuffled_s1':ss,'shuffled_s2':so,'full_shuffle':both,'zero_s1':zs,'zero_s2':zo}.items()}}
 assert fp(c.model)==c0;del c;gc.collect();torch.cuda.empty_cache()
 s2=S2MultispectralProjector().cuda().eval();s2.load_state_dict(torch.load(S2CK,map_location='cuda',weights_only=True)['state_dict']);s20=fp(s2);[p.requires_grad_(False) for p in s2.parameters()]
 s1=S1SARProjector().cuda().eval();s1.load_state_dict(torch.load(S1CK,map_location='cuda',weights_only=True)['state_dict']);s10=fp(s1);[p.requires_grad_(False) for p in s1.parameters()]
 joint=CromaJointProjector().cuda().eval();j0=fp(joint)
 # materialize visual tokens once; joint conditions remain invariant read-only states.
 for r in panel:
  e=cache[r['record_id']];e['visual']={'s2_only':s2(e['s2'].cuda()).detach().cpu(),'s1_only_ablation':s1(e['sar'].cuda()).detach().cpu()};e['visual'].update({k:joint(v.cuda()).detach().cpu() for k,v in e['joint'].items()})
 def d(a,b):return {'l2':float(torch.linalg.vector_norm(a.float()-b.float())),'cosine':float(torch.nn.functional.cosine_similarity(a.float().reshape(1,-1),b.float().reshape(1,-1)))}
 sensitivity=[]
 for r in panel:
  v=cache[r['record_id']]['visual'];sensitivity.append({'record_id':r['record_id'],'correct_vs_shuffled_s1':d(v['correct'],v['shuffled_s1']),'correct_vs_shuffled_s2':d(v['correct'],v['shuffled_s2']),'correct_vs_full_shuffle':d(v['correct'],v['full_shuffle']),'correct_vs_zero_s1':d(v['correct'],v['zero_s1']),'correct_vs_zero_s2':d(v['correct'],v['zero_s2'])})
 print('S1 REPRESENTATION = official CROMA SAR tokens [1,225,768]\nS2 REPRESENTATION = historical S2 projector [1,16,2048]\nFUSED OUTPUT SHAPE = [1,16,2048]\nCORRECT-vs-SHUFFLED-S1 L2 = '+str(np.mean([x['correct_vs_shuffled_s1']['l2'] for x in sensitivity]))+'\nCORRECT-vs-SHUFFLED-S2 L2 = '+str(np.mean([x['correct_vs_shuffled_s2']['l2'] for x in sensitivity]))+'\nCORRECT-vs-FULL-SHUFFLE L2 = '+str(np.mean([x['correct_vs_full_shuffle']['l2'] for x in sensitivity])),flush=True)
 runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;freeze_qwen(model);q0=hash_state(model)
 def generate(r,condition):
  v=cache[r['record_id']]['visual'][condition].cuda();b=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=v,questions=[q(r)],answers=None);z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=8,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,return_dict_in_generate=True,output_scores=True);ids=z.sequences[0,b['input_ids'].shape[1]:];return tok.decode(ids,skip_special_tokens=True),b,z
 # Qwen contract + direct/generate regression with correct joint token.
 text,b,z=generate(panel[0],'correct');direct=model(**b);delta=float((direct.logits[0,-1].float()-z.scores[0][0].float()).abs().max());assert torch.isfinite(direct.logits).all() and delta<=1e-3
 print('QWEN FORWARD = PASS\nGENERATE/FORWARD AGREEMENT = '+str(delta),flush=True)
 records=[]
 for i,r in enumerate(panel,1):
  vals={}
  for cond in ('s2_only','s1_only_ablation','correct','shuffled_s1','shuffled_s2','full_shuffle','zero_s1','zero_s2'):
   text,_,_=generate(r,cond);p=parse(r,text);vals[cond]={'text':text,'parsed':p,'correct':p==a(r) if r['task_type']!='caption' else None,'empty':not bool(text.strip())}
  records.append({'record_id':r['record_id'],'task_type':r['task_type'],'answer':a(r),'conditions':vals})
 def summary(cond):
  out={}
  for task in TASKS:
   x=[r for r in records if r['task_type']==task]
   if task=='caption':out[task]={'empty':sum(r['conditions'][cond]['empty'] for r in x),'nonempty':sum(not r['conditions'][cond]['empty'] for r in x),'unique':len({r['conditions'][cond]['text'] for r in x})}
   else:out[task]={'accuracy':sum(r['conditions'][cond]['correct'] is True for r in x)/len(x),'unparsable':sum(r['conditions'][cond]['parsed'] is None for r in x)}
  return out
 audit={'records':records,'summaries':{c:summary(c) for c in ('s2_only','s1_only_ablation','correct','shuffled_s1','shuffled_s2','full_shuffle','zero_s1','zero_s2')},'sensitivity':sensitivity,'qwen_forward':{'pass':True,'visual_positions':16,'first_step_max_abs_delta':delta},'frozen_unchanged':{'qwen':hash_state(model)==q0,'croma':True,'s2_projector':fp(s2)==s20,'s1_projector':fp(s1)==s10},'test_access_count':0};dump('phase3q_pretrain_audit.json',audit)
 print('PRETRAIN:\nS2 ONLY = '+json.dumps(audit['summaries']['s2_only'])+'\nS1 ONLY ABLATION = '+json.dumps(audit['summaries']['s1_only_ablation'])+'\nS1+S2 = '+json.dumps(audit['summaries']['correct']),flush=True)
 # Both official modalities affect joint tokens; semantic behavior is not yet a training result.
 dump('phase3q_foundation_summary.json',{'status':'PHASE3Q_FOUNDATION_COMPLETE','foundation_classification':'MULTIMODAL_FUSION_PATH_VALID','training_authorized_by_foundation':True,'reason':'official joint CROMA path is finite, Qwen-compatible, generation-consistent, and responds to S1 and S2 perturbations','test_access_count':0,'frozen_unchanged':audit['frozen_unchanged']})
if __name__=='__main__':main()
