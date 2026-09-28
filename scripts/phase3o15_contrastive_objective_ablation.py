"""Strictly bounded Phase 3O.15 objective ablation; never trains historical state."""
from __future__ import annotations

import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.phase3o11_postfix_semantic_audit import cube, hash_state
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.image_conditioned_objective import image_conditioned_penalty, target_sequence_log_probability
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.training import build_qwen_token_batch, freeze_qwen, optimizer_parameter_ids
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

RUN12=ROOT/'artifacts/training/phase3o/phase3o12_run'; OUT=ROOT/'artifacts/training/phase3o/phase3o15_objective_ablation'; CK=RUN12/'phase3o12_projector_final.pt'
SHA='5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0'; FP='168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b'; REV='66285546d2b821cf421d4f5eb2576359d3770cd3'
SNAPSHOT=Path(r'C:\Users\Rishab\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots')/REV; TASKS=('binary_qa','multiple_choice_qa','caption')
VARIANTS={
 'A_BASE_CE':{'kind':'none','weight':0.0,'margin':0.0,'caption_only':False},
 'B_HINGE_025':{'kind':'hinge','weight':0.25,'margin':0.25,'caption_only':False},
 'C_HINGE_010':{'kind':'hinge','weight':0.10,'margin':0.25,'caption_only':False},
 'D_SMOOTH_LL':{'kind':'softplus','weight':0.25,'margin':0.0,'caption_only':False},
 'E_CAPTION_SMOOTH_LL':{'kind':'softplus','weight':0.25,'margin':0.0,'caption_only':True},
}
PREREG={'phase':'3O.15','seed':3015,'initial_checkpoint_sha256':SHA,'initial_fingerprint':FP,'train_counts':{t:8 for t in TASKS},'validation_counts':{t:4 for t in TASKS},'steps_per_variant':8,'microbatch_order':'record_id sorted, paired by task index: each of 8 steps contains one Binary, one MCQ, one Caption example','optimizer':{'name':'AdamW','learning_rate':0.0001,'weight_decay':0.0001},'precision':'float16','test_access_count':0,'variants':VARIANTS,'decision_rule':'PROMISING only when all validation task margins do not materially worsen (tolerance -0.001), at least two improve by >0.001, caption diversity does not decrease, losses are finite, projector gradients are nonzero, Qwen gradients are zero, and integrity passes.'}

def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
def req(x,m):
 if not x: raise RuntimeError('PHASE3O15_BLOCKED: '+m)
def fp(state):
 p=S2MultispectralProjector();p.load_state_dict(state);return p.fingerprint()
def ans(r): return r['caption'] if r['task_type']=='caption' else r['answer']
def question(r): return 'Describe this Sentinel-2 image.' if r['task_type']=='caption' else r['question']
def index():
 x={}
 for root in (Path(r'D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2'),Path(r'D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2')):
  for p in root.rglob('*.tif'):
   if '_B' in p.stem:x.setdefault(p.name,p)
 return x
def getasset(r,ix,cache):
 if r['record_id'] not in cache:cache[r['record_id']]=inspect_area(r,{b:ix[f"{r['image_id']}_{b}.tif"] for b in S2_BANDS}).to_dict()
 return cache[r['record_id']]
def mapping(rows,split):
 result=[]
 for task in TASKS:
  group=sorted((r for r in rows if r['task_type']==task),key=lambda r:(r['image_id'],r['record_id']))
  for i,r in enumerate(group):
   peer=next(q for q in group[i+1:]+group[:i+1] if q['image_id']!=r['image_id'])
   result.append({'sample_id':r['record_id'],'task_type':task,'split':split,'source_image_id':r['image_id'],'shuffled_sample_id':peer['record_id'],'shuffled_image_id':peer['image_id']})
 return {x['sample_id']:x for x in result}
def pair_outputs(model,tok,proj,row,wrong,ix,cache):
 a=build_qwen_token_batch(tokenizer=tok,model=model,projector=proj,s2=cube(getasset(row,ix,cache)),questions=[question(row)],answers=[ans(row)])
 b=build_qwen_token_batch(tokenizer=tok,model=model,projector=proj,s2=cube(getasset(wrong,ix,cache)),questions=[question(row)],answers=[ans(row)])
 return a,b,model(**a),model(**b)
def margin_audit(model,tok,proj,rows,maps,lookup,ix,cache):
 es=[]
 with torch.inference_mode():
  for r in rows:
   _,b,co,so=pair_outputs(model,tok,proj,r,lookup[maps[r['record_id']]['shuffled_sample_id']],ix,cache)
   cs=target_sequence_log_probability(co.logits, b['labels']); ss=target_sequence_log_probability(so.logits,b['labels'])
   es.append({'sample_id':r['record_id'],'task_type':r['task_type'],'correct_score':float(cs),'shuffled_score':float(ss),'margin':float(cs-ss)})
 means={t:sum(e['margin'] for e in es if e['task_type']==t)/sum(e['task_type']==t for e in es) for t in TASKS}
 return {'entries':es,'means':means}
def gen(model,tok,proj,r,ix,cache):
 b=build_qwen_token_batch(tokenizer=tok,model=model,projector=proj,s2=cube(getasset(r,ix,cache)),questions=[question(r)],answers=None)
 z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=16,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0,use_cache=True)
 return tok.decode(z[0,b['input_ids'].shape[1]:],skip_special_tokens=True)
def output_audit(model,tok,proj,rows,maps,lookup,ix,cache):
 vals=[]
 with torch.inference_mode():
  for r in rows:
   normal=gen(model,tok,proj,r,ix,cache);wrong=lookup[maps[r['record_id']]['shuffled_sample_id']]; shuffled=gen(model,tok,proj,wrong,ix,cache)
   vals.append({'sample_id':r['record_id'],'task_type':r['task_type'],'correct_image_output':normal,'shuffled_image_output':shuffled,'changed_under_shuffle':normal!=shuffled})
 result={}
 for task in TASKS:
  x=[v for v in vals if v['task_type']==task]; texts=[v['correct_image_output'] for v in x]
  result[task]={'count':len(x),'changed_under_shuffle':sum(v['changed_under_shuffle'] for v in x),'empty':sum(not t.strip() for t in texts),'unique':len(set(texts)),'largest_mode':Counter(texts).most_common(1)[0][1]}
 return {'records':vals,'summary':result}
def main():
 print('PHASE3O15 START',flush=True);req(not OUT.exists(),'namespace exists');req(sha256_file(CK)==SHA,'checkpoint SHA');state=torch.load(CK,map_location='cpu',weights_only=False)['state_dict'];req(fp(state)==FP and SNAPSHOT.is_dir(),'checkpoint fingerprint or Qwen revision')
 manifest=json.loads((RUN12/'phase3o12_training_manifest.json').read_text());req(manifest['test_access_count']==0,'TEST access'); train=manifest['selection']['train']; val=manifest['selection']['validation']
 panels={}
 for split,rows,n in [('train',train,8),('validation',val,4)]:
  panels[split]=[r for t in TASKS for r in sorted((x for x in rows if x['task_type']==t),key=lambda x:x['record_id'])[:n]]
 req({t:sum(r['task_type']==t for r in panels['train']) for t in TASKS}==PREREG['train_counts'],'train panel');req({t:sum(r['task_type']==t for r in panels['validation']) for t in TASKS}==PREREG['validation_counts'],'validation panel')
 req(not {r['image_id'] for r in panels['train']}&{r['image_id'] for r in panels['validation']},'panel image overlap'); maps={'train':mapping(panels['train'],'train'),'validation':mapping(panels['validation'],'validation')};req(all(x['source_image_id']!=x['shuffled_image_id'] for m in maps.values() for x in m.values()),'self image mapping')
 OUT.mkdir(parents=True);dump(OUT/'phase3o15_preregistration.json',PREREG);dump(OUT/'phase3o15_train_panel.json',{'records':panels['train'],'test_access_count':0});dump(OUT/'phase3o15_validation_panel.json',{'records':panels['validation'],'test_access_count':0});dump(OUT/'phase3o15_train_shuffle_map.json',{'mapping':list(maps['train'].values()),'test_access_count':0});dump(OUT/'phase3o15_validation_shuffle_map.json',{'mapping':list(maps['validation'].values()),'test_access_count':0})
 ix=index();cache={}; lookup={'train':{r['record_id']:r for r in panels['train']},'validation':{r['record_id']:r for r in panels['validation']}}; runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda'); model=runtime['model'].eval();tok=runtime['processor'].tokenizer;req(freeze_qwen(model)['qwen_trainable_parameter_count']==0,'Qwen freeze');qbefore=hash_state(model)
 results={}; reference_pre=None
 for name,spec in VARIANTS.items():
  proj=S2MultispectralProjector().to('cuda');proj.load_state_dict(state);proj.train();req(fp(proj.state_dict())==FP,'variant initialization')
  pretrain=margin_audit(model,tok,proj,panels['train'],maps['train'],lookup['train'],ix,cache);preval=margin_audit(model,tok,proj,panels['validation'],maps['validation'],lookup['validation'],ix,cache); pregen=output_audit(model,tok,proj,panels['validation'],maps['validation'],lookup['validation'],ix,cache)
  baseline={'train':pretrain['means'],'validation':preval['means']}
  if reference_pre is None:reference_pre=baseline
  else:req(all(abs(baseline[k][t]-reference_pre[k][t])<1e-6 for k in baseline for t in TASKS),'nonidentical pre-margin')
  opt=torch.optim.AdamW(proj.parameters(),lr=1e-4,weight_decay=1e-4);req(not optimizer_parameter_ids(opt).intersection(id(p) for p in model.parameters()),'optimizer Qwen membership');steps=[]
  ordered={t:sorted((r for r in panels['train'] if r['task_type']==t),key=lambda r:r['record_id']) for t in TASKS}
  for step in range(8):
   opt.zero_grad(set_to_none=True);proj.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True); totals=defaultdict(float)
   for task in TASKS:
    r=ordered[task][step]; _,b,co,so=pair_outputs(model,tok,proj,r,lookup['train'][maps['train'][r['record_id']]['shuffled_sample_id']],ix,cache); cs=target_sequence_log_probability(co.logits,b['labels']);ss=target_sequence_log_probability(so.logits,b['labels']); margin=cs-ss
    kind='none' if spec['caption_only'] and task!='caption' else spec['kind']; penalty=image_conditioned_penalty(margin,kind=kind,weight=spec['weight'],margin=spec['margin']); loss=co.loss+penalty; req(torch.isfinite(loss),'nonfinite loss');(loss/3).backward()
    for key,value in {'base_ce':co.loss,'contrastive':penalty,'total':loss,'correct_score':cs.mean(),'shuffled_score':ss.mean(),'margin':margin.mean()}.items():totals[key]+=float(value.detach())/3
   grads=[p.grad for p in proj.parameters()];gn=math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None));qg=sum(p.grad is not None for p in model.parameters());req(gn>0 and qg==0,'gradient violation');opt.step();steps.append({'step':step+1,**dict(totals),'projector_grad_norm':gn,'qwen_gradient_tensors':qg,'finite':True})
  proj.eval();posttrain=margin_audit(model,tok,proj,panels['train'],maps['train'],lookup['train'],ix,cache);postval=margin_audit(model,tok,proj,panels['validation'],maps['validation'],lookup['validation'],ix,cache);postgen=output_audit(model,tok,proj,panels['validation'],maps['validation'],lookup['validation'],ix,cache)
  delta={t:postval['means'][t]-preval['means'][t] for t in TASKS}; capok=postgen['summary']['caption']['unique']>=pregen['summary']['caption']['unique']; improves=sum(x>0.001 for x in delta.values()); notworse=all(x>=-0.001 for x in delta.values()); klass='PROMISING' if improves>=2 and notworse and capok else 'CAPTION_DEGENERATES' if not capok else 'TRAIN_ONLY_EFFECT' if any(posttrain['means'][t]-pretrain['means'][t]>0.001 for t in TASKS) else 'NO_EFFECT'
  results[name]={'specification':spec,'initial_fingerprint':FP,'pre':{'train':pretrain,'validation':preval,'generation':pregen},'steps':steps,'post':{'train':posttrain,'validation':postval,'generation':postgen},'validation_margin_delta':delta,'classification':klass,'projector_gradients_nonzero':True,'qwen_gradient_tensors':0};print(f'VARIANT = {name}\nBINARY PRE TRAIN MARGIN = {pretrain["means"]["binary_qa"]}\nBINARY POST TRAIN MARGIN = {posttrain["means"]["binary_qa"]}\nBINARY VALIDATION DELTA = {delta["binary_qa"]}\nMCQ PRE TRAIN MARGIN = {pretrain["means"]["multiple_choice_qa"]}\nMCQ POST TRAIN MARGIN = {posttrain["means"]["multiple_choice_qa"]}\nMCQ VALIDATION DELTA = {delta["multiple_choice_qa"]}\nCAPTION PRE TRAIN MARGIN = {pretrain["means"]["caption"]}\nCAPTION POST TRAIN MARGIN = {posttrain["means"]["caption"]}\nCAPTION VALIDATION DELTA = {delta["caption"]}\nCAPTION UNIQUE PRE = {pregen["summary"]["caption"]["unique"]}\nCAPTION UNIQUE POST = {postgen["summary"]["caption"]["unique"]}\nQWEN GRADIENTS = 0\nVARIANT CLASSIFICATION = {klass}',flush=True)
 req(hash_state(model)==qbefore and sha256_file(CK)==SHA and fp(torch.load(CK,map_location='cpu',weights_only=False)['state_dict'])==FP,'historical immutability'); promising=[n for n,v in results.items() if v['classification']=='PROMISING'];artifact={'phase':'3O.15','status':'PHASE3O15_COMPLETE','preregistration':PREREG,'test_access_count':0,'shared_initial_fingerprint':FP,'results':results,'qwen_unchanged':True,'phase3o12_projector_unchanged':True,'best_justified_variant':promising[0] if len(promising)==1 else None,'overall_conclusion':'OBJECTIVE_PROMISING' if promising else 'NO_VARIANT_JUSTIFIED_FOR_FULL_TRAINING'};dump(OUT/'phase3o15_variant_results.json',artifact);print('BEST JUSTIFIED VARIANT = '+(artifact['best_justified_variant'] or 'NO VARIANT JUSTIFIED FOR FULL TRAINING'),flush=True)
if __name__=='__main__':main()
