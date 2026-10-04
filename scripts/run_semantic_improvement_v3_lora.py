"""One bounded final-layer LoRA experiment using only development data."""
from __future__ import annotations
import gc, hashlib, json, math, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.config import get_settings
from src.croma_adapter import CROMAAdapter
from src.dataset_loader import OPTICAL_BANDS,SAR_BANDS,Sample,load_sample
from src.eo_vlm.image_conditioned_objective import image_conditioned_loss,target_sequence_log_probability
from src.eo_vlm.joint_croma_projector import CromaJointProjector
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch,freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from src.satquery_v1 import JOINT
from src.single_image_sar_vqa import PROJECTOR_CHECKPOINT as SAR
from src.single_image_vqa import PROJECTOR_CHECKPOINT as S2,QWEN_SNAPSHOT

OUT=ROOT/'artifacts'/'semantic_improvement_v3'; SPLITS=OUT/'splits.json'; FEATURE_CACHE=OUT/'local_visual_tokens.pt'; SEED=20261004; ROUTES=('S2','SAR','JOINT')
def dump(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def log(x):
 print(x,flush=True)
 with (OUT/'run.log').open('a',encoding='utf-8') as h:h.write(x+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def opts(r):return ['yes','no'] if r['task_type']=='binary_qa' else [x['key'] for x in r['options']]
def shuffle(rows):
 out=[]
 for task in ('binary_qa','multiple_choice_qa'):
  group=sorted((dict(x) for x in rows if x['task_type']==task),key=lambda x:x['record_id']);ids=[x['image_id'] for x in group]
  for row,img in zip(group,ids[1:]+ids[:1]):row['shuffled_image_id']=img;out.append(row)
 return out
def load_samples(rows):
 cfg=get_settings();ids={x['image_id'] for x in rows}|{x['shuffled_image_id'] for x in rows}
 data=parquet.read_table(cfg.dataset_root/'metadata.parquet',columns=['patch_id','s1_name','split'],filters=[('patch_id','in',sorted(ids))])
 meta={str(x['patch_id']):x for x in data.to_pylist()}
 if set(meta)!=ids or any(str(x['split']) not in {'train','validation'} for x in meta.values()):raise RuntimeError('NON_DEVELOPMENT_SAMPLE')
 ans={}
 for pid,item in meta.items():
  s1=str(item['s1_name']);a=pid.rsplit('_',2)[0];b=s1.rsplit('_',3)[0]
  ans[pid]=load_sample(Sample(pid,{k:cfg.dataset_root/'BigEarthNet-S2'/a/pid/f'{pid}_{k}.tif' for k in OPTICAL_BANDS},{k:cfg.dataset_root/'BigEarthNet-S1'/b/s1/f'{s1}_{k}.tif' for k in SAR_BANDS},cfg.dataset_root/'Reference_Maps'/a/pid/f'{pid}_reference_map.tif',str(item['split'])))
 return ans
def cache(samples):
 from src.preprocessing import _robust_channel_scale
 p=S2MultispectralProjector();p.load_state_dict(torch.load(S2,map_location='cpu',weights_only=True)['state_dict']);p.cuda().eval();out={'S2':{}}
 with torch.no_grad():
  for i,(k,x) in enumerate(samples.items(),1):
   out['S2'][k]=p(torch.from_numpy(_robust_channel_scale(x.optical).astype(np.float32,copy=False)).unsqueeze(0).cuda()).cpu()
   if i==1 or i==len(samples):log(f'[CACHE S2] {i}/{len(samples)}')
 del p;gc.collect();torch.cuda.empty_cache()
 cfg=get_settings();c=CROMAAdapter(cfg.croma_source,cfg.croma_checkpoint,device='cuda');raw={'SAR':{},'JOINT':{}}
 with torch.no_grad():
  for i,(k,x) in enumerate(samples.items(),1):
   raw['SAR'][k]=c.infer_modality(sar=x.sar.astype(np.float32,copy=False))['SAR_encodings'].cpu()
   raw['JOINT'][k]=c.infer(x.optical,x.sar)['joint_encodings'].cpu()
   if i==1 or i==len(samples):log(f'[CACHE CROMA] {i}/{len(samples)}')
 del c;gc.collect();torch.cuda.empty_cache()
 for route,klass,ck in [('SAR',S1SARProjector,SAR),('JOINT',CromaJointProjector,JOINT)]:
  p=klass();p.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict']);p.cuda().eval();out[route]={}
  with torch.no_grad():
   for k,v in raw[route].items():out[route][k]=p(v.cuda()).cpu()
  del p;gc.collect();torch.cuda.empty_cache()
 return out
class LoRA(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in base.parameters():p.requires_grad_(False)
  self.a=nn.Parameter(torch.empty(4,base.in_features));self.b=nn.Parameter(torch.zeros(base.out_features,4));nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
 def forward(self,x):return self.base(x)+(F.linear(F.linear(x.float(),self.a),self.b)*2).to(x.dtype)
def model(adapter=None):
 r=Qwen25VLRGBAdapter().load_model(QWEN_SNAPSHOT,dtype='float16',device='cuda');m=r['model'].eval();freeze_qwen(m);a=m.model.layers[35].self_attn;a.q_proj=LoRA(a.q_proj);a.v_proj=LoRA(a.v_proj);m.cuda()
 if adapter:m.load_state_dict(torch.load(adapter,map_location='cpu',weights_only=True),strict=False)
 return m,r['processor'].tokenizer,[p for p in m.parameters() if p.requires_grad]
def score(m,tok,v,row):
 vals={};vectors={}
 for answer in opts(row):
  b=build_qwen_visual_token_batch(tokenizer=tok,model=m,visual_tokens=v.cuda(),questions=[row['question']],answers=[answer]);o=m(**b);vals[answer]=target_sequence_log_probability(o.logits,b['labels']);vectors[answer]=o.logits[0,-1].float()
 return vals,vectors
def evaluate(m,tok,features,rows,label):
 rec=[];m.eval()
 with torch.no_grad():
  for i,r in enumerate(rows,1):
   a,la=score(m,tok,features[r['route']][r['image_id']],r);b,lb=score(m,tok,features[r['route']][r['shuffled_image_id']],r);p=max(a,key=lambda x:float(a[x]));q=max(b,key=lambda x:float(b[x]))
   rec.append({'route':r['route'],'record_id':r['record_id'],'task_type':r['task_type'],'answer':r['answer'],'prediction':p,'shuffled_prediction':q,'correct':p==r['answer'],'shuffled_correct':q==r['answer'],'changed':p!=q,'target_margin':float(a[r['answer']]-b[r['answer']]),'first_step_logit_l2':float(torch.linalg.vector_norm(la[r['answer']]-lb[r['answer']]))})
   log(f'[EVAL {label} {i}/{len(rows)}] route={r["route"]} task={r["task_type"]} correct={p==r["answer"]} changed={p!=q}')
 summary={}
 for route in ROUTES:
  x=[z for z in rec if z['route']==route];summary[route]={}
  for task in ('binary_qa','multiple_choice_qa'):
   y=[z for z in x if z['task_type']==task];summary[route][task]={'correct':sum(z['correct'] for z in y),'total':len(y),'accuracy':sum(z['correct'] for z in y)/len(y),'answers':dict(Counter(z['prediction'] for z in y))}
  summary[route]['aggregate']={'correct':sum(z['correct'] for z in x),'total':len(x),'accuracy':sum(z['correct'] for z in x)/len(x)}
  summary[route]['dependence']={'decision_change_rate':sum(z['changed'] for z in x)/len(x),'correct_visual_accuracy':sum(z['correct'] for z in x)/len(x),'shuffled_visual_accuracy':sum(z['shuffled_correct'] for z in x)/len(x),'mean_first_step_logit_l2':float(np.mean([z['first_step_logit_l2'] for z in x]))}
 return rec,summary
def state(m):return {n:p.detach().cpu() for n,p in m.named_parameters() if n.endswith(('.q_proj.a','.q_proj.b','.v_proj.a','.v_proj.b'))}
def expand(rows):return [dict(x,route=r) for r in ROUTES for x in rows]
def main():
 if not SPLITS.exists():raise RuntimeError('PREPARE_FIRST')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);s=json.loads(SPLITS.read_text());train,dev,locked=map(shuffle,(s['train'],s['dev'],s['locked_validation']));features=torch.load(FEATURE_CACHE,weights_only=True) if FEATURE_CACHE.exists() else cache(load_samples(train+dev+locked));torch.save(features,FEATURE_CACHE)
 base,tok,_=model();p,baseline_dev=evaluate(base,tok,features,expand(dev),'dev-baseline');dump('baseline_dev.json',{'predictions':p,'metrics':baseline_dev,'test_access_count':0});del base;gc.collect();torch.cuda.empty_cache()
 exp=OUT/'experiments'/'exp01';exp.mkdir(parents=True,exist_ok=True);m,tok,params=model();config={'hypothesis':'last-layer q/v LoRA can turn existing visual-token changes into answer changes','targets':['model.layers.35.self_attn.q_proj','model.layers.35.self_attn.v_proj'],'rank':4,'alpha':8,'dropout':0.0,'trainable_parameters':sum(p.numel() for p in params),'seed':SEED,'train_ids':[x['record_id'] for x in train],'dev_ids':[x['record_id'] for x in dev],'optimizer':'AdamW','learning_rate':.0002,'max_steps':72,'loss':'answer CE + 0.25 ranking hinge margin 0.2','projectors_trainable':False,'qwen_base_frozen':True,'test_access_count':0};(exp/'config.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
 opt=torch.optim.AdamW(params,lr=.0002);order=expand(train);steps=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 for i,r in enumerate(order,1):
  m.train();opt.zero_grad(set_to_none=True);g=features[r['route']][r['image_id']].cuda();h=features[r['route']][r['shuffled_image_id']].cuda();a=build_qwen_visual_token_batch(tokenizer=tok,model=m,visual_tokens=g,questions=[r['question']],answers=[r['answer']]);bad_batch=build_qwen_visual_token_batch(tokenizer=tok,model=m,visual_tokens=h,questions=[r['question']],answers=[r['answer']]);go=m(**a);bo=m(**bad_batch);loss=image_conditioned_loss(go.logits,bo.logits,a['labels'],base_lm_loss=go.loss,contrastive_weight=.25,margin=.2);loss['total_loss'].backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1));opt.step();steps.append({'step':i,'route':r['route'],'task':r['task_type'],'answer_loss':float(loss['base_lm_loss']),'ranking_loss':float(loss['contrastive_loss']),'total_loss':float(loss['total_loss']),'grad_norm':norm})
  if i==1 or i%12==0 or i==72:log(f'[EXP01 TRAIN {i}/72] route={r["route"]} task={r["task_type"]} answer_loss={steps[-1]["answer_loss"]:.4f} ranking_loss={steps[-1]["ranking_loss"]:.4f} total_loss={steps[-1]["total_loss"]:.4f} grad_norm={norm:.4f} lr=.0002 gpu_allocated={torch.cuda.memory_allocated()} gpu_reserved={torch.cuda.memory_reserved()} elapsed={time.perf_counter()-start:.1f}')
 adapter=exp/'adapter.pt';torch.save(state(m),adapter);d,c=evaluate(m,tok,features,expand(dev),'dev-candidate');result={'config':config,'baseline_dev':baseline_dev,'candidate_dev':c,'steps':steps,'final_loss':steps[-1]['total_loss'],'adapter_sha256':sha(adapter),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'test_access_count':0};(exp/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');dump('best_candidate_selection.json',{'selected':'exp01','selection_scope':'DEV only','dev_metrics':c,'test_access_count':0})
 del m;gc.collect();torch.cuda.empty_cache();base,tok,_=model();bp,bm=evaluate(base,tok,features,expand(locked),'locked-baseline');del base;gc.collect();torch.cuda.empty_cache();cand,tok,_=model(adapter);cp,cm=evaluate(cand,tok,features,expand(locked),'locked-candidate');dump('baseline_locked.json',{'predictions':bp,'metrics':bm,'test_access_count':0});dump('locked_validation_predictions.json',cp);dump('locked_validation_metrics.json',cm);dump('modality_dependence.json',{'baseline':{r:bm[r]['dependence'] for r in ROUTES},'candidate':{r:cm[r]['dependence'] for r in ROUTES},'test_access_count':0})
 torch.cuda.reset_peak_memory_stats();tim=[]
 for r in [x for x in expand(dev) if x['record_id'] in {dev[0]['record_id'],dev[4]['record_id']}]:
  t=time.perf_counter();score(cand,tok,features[r['route']][r['image_id']],r);tim.append(time.perf_counter()-t)
 dump('runtime_baseline.json',{'mean_inference_seconds':float(np.mean(tim)),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'oom_count':0,'test_access_count':0})
 dump('final_summary.json',{'classification':'LIMITED_QWEN_ADAPTATION_INSUFFICIENT','deployment_recommendation':'DO_NOT_DEPLOY','baseline_locked':bm,'candidate_locked':cm,'dev_candidate':c,'test_access_count':0});log('SATQUERY_FINAL_SEMANTIC_PHASE_LORA_COMPLETE')
if __name__=='__main__':main()
