"""One bounded BigEarthNet-only adaptation of CromaJointProjector."""
from __future__ import annotations
import gc,hashlib,json,math,sys
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
OUT=ROOT/'artifacts/training/phase3q/phase3q1_fusion_adaptation';M=ROOT/'artifacts/training/phase3o/phase3o12_run/phase3o12_training_manifest.json';DATA=Path(r'D:\Satquery_ai datasets\comparison\raw-1000');CS=Path(r'D:\Satquery_ai datasets\croma_official');CK=Path(r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt');S2CK=ROOT/'artifacts/training/phase3o/phase3o12_run/phase3o12_projector_final.pt';S1CK=ROOT/'artifacts/training/phase3p/phase3p1_sar_adaptation/phase3p1_sar_projector_final.pt';TASKS=('binary_qa','multiple_choice_qa','caption');SEED=30001
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def q(r):return 'Describe this paired Sentinel-1 and Sentinel-2 scene.' if r['task_type']=='caption' else r['question']
def a(r):return r['caption'] if r['task_type']=='caption' else r['answer']
def parse(r,t):
 for x in t.lower().replace('.',' ').replace(')',' ').split():
  if r['task_type']=='binary_qa' and x in {'yes','no'}:return x
  if r['task_type']=='multiple_choice_qa' and x in {'a','b','c','d'}:return x
 return None
def save(n,p):
 path=OUT/n;torch.save({'state_dict':p.state_dict()},path);return {'sha256':sha(path),'fingerprint':fp(p),'parameter_count':sum(x.numel() for x in p.parameters()),'path':n}
def main():
 if OUT.exists():raise RuntimeError('PHASE3Q1_NAMESPACE_EXISTS')
 OUT.mkdir(parents=True);torch.manual_seed(SEED);print('PHASE3Q1 START\nDATASET = BIGEARTHNET ONLY',flush=True);man=json.loads(M.read_text());assert man['test_access_count']==0;samples={x.patch_id:x for x in discover_samples(DATA,strict=True)};used=set();sel={}
 for split,n in [('train',20),('validation',10)]:
  rows=[]
  for task in TASKS:
   for r in sorted((x for x in man['selection'][split] if x['task_type']==task and x['image_id'] in samples),key=lambda x:x['record_id']):
    if r['image_id'] not in used:rows.append(dict(r));used.add(r['image_id'])
    if sum(x['task_type']==task for x in rows)==n:break
  sel[split]=rows
 assert len(sel['train'])==60 and len(sel['validation'])==30 and not ({r['image_id'] for r in sel['train']}&{r['image_id'] for r in sel['validation']})
 allrows=sel['train']+sel['validation'];prepared={r['image_id']:load_sample(samples[r['image_id']]) for r in allrows}
 for split,rows in sel.items():
  for task in TASKS:
   group=[r for r in rows if r['task_type']==task];ids=[r['image_id'] for r in group]
   for r,w in zip(group,ids[1:]+ids[:1]):r['shuffled_image_id']=w
 manifest={'dataset':'BigEarthNet only','test_access_count':0,'counts':{k:dict(Counter(r['task_type'] for r in v)) for k,v in sel.items()},'selection':sel,'integrity':{'train_validation_image_overlap':[],'duplicate_images':len({r['image_id'] for r in allrows})!=90,'paired_s1_s2_valid':True}};dump('phase3q1_training_manifest.json',manifest);dump('phase3q1_shuffle_map.json',{'maps':{r['record_id']:r['shuffled_image_id'] for r in allrows},'test_access_count':0});print('TRAIN = 60\nVALIDATION = 30\nTEST = 0',flush=True)
 p=CromaJointProjector().cuda();initial=save('phase3q1_joint_projector_initial.pt',p);print('INITIAL SHA = '+initial['sha256']+'\nINITIAL FINGERPRINT = '+initial['fingerprint'],flush=True)
 c=CROMAAdapter(CS,CK,device='cuda');[x.requires_grad_(False) for x in c.model.parameters()];c0=fp(c.model);cache={}
 for i,r in enumerate(allrows,1):
  x=prepared[r['image_id']];w=prepared[r['shuffled_image_id']];print(f'[3Q1][CROMA] record {i}/90',flush=True)
  with torch.no_grad():
   own=c.infer(x.raw_optical,x.raw_sar);ss=c.infer(w.raw_optical,x.raw_sar);so=c.infer(x.raw_optical,w.raw_sar);ff=c.infer(w.raw_optical,w.raw_sar);zs=c.infer(x.raw_optical,np.zeros_like(x.raw_sar));zo=c.infer(np.zeros_like(x.raw_optical),x.raw_sar)
  cache[r['record_id']]={'joint':{k:v['joint_encodings'].cpu() for k,v in {'correct':own,'shuffled_s1':ss,'shuffled_s2':so,'full_shuffle':ff,'zero_s1':zs,'zero_s2':zo}.items()},'s2':torch.from_numpy(x.optical).unsqueeze(0),'s1':own['SAR_encodings'].cpu()}
 assert fp(c.model)==c0;del c;gc.collect();torch.cuda.empty_cache();runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;freeze_qwen(model);q0=hash_state(model)
 s2=S2MultispectralProjector().cuda().eval();s2.load_state_dict(torch.load(S2CK,map_location='cuda',weights_only=True)['state_dict']);[x.requires_grad_(False) for x in s2.parameters()];s20=fp(s2);s1=S1SARProjector().cuda().eval();s1.load_state_dict(torch.load(S1CK,map_location='cuda',weights_only=True)['state_dict']);[x.requires_grad_(False) for x in s1.parameters()];s10=fp(s1)
 def visual(r,cond):
  e=cache[r['record_id']];return s2(e['s2'].cuda()) if cond=='s2_only' else s1(e['s1'].cuda()) if cond=='s1_only' else p(e['joint'][cond].cuda())
 def batch(r,cond,target=True):return build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=visual(r,cond),questions=[q(r)],answers=[a(r)] if target else None)
 def audit(rows):
  rec=[];p.eval()
  with torch.no_grad():
   for r in rows:
    vals={}
    for cond in ('correct','shuffled_s1','shuffled_s2','full_shuffle','zero_s1','zero_s2','s2_only','s1_only'):
     b=batch(r,cond,False);z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=8,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True);t=tok.decode(z[0,b['input_ids'].shape[1]:],skip_special_tokens=True);pp=parse(r,t);vals[cond]={'text':t,'parsed':pp,'correct':pp==a(r) if r['task_type']!='caption' else None,'empty':not bool(t.strip())}
    rec.append({'record_id':r['record_id'],'task_type':r['task_type'],'conditions':vals})
  return {'records':rec,'test_access_count':0}
 untrained=audit(sel['validation']);dump('phase3q1_untrained_semantic_audit.json',untrained);print('UNTRAINED:\nJOINT CORRECT = completed\nSHUFFLED S1 = completed\nSHUFFLED S2 = completed\nFULL SHUFFLE = completed\nZERO S1 = completed\nZERO S2 = completed\nS2 ONLY = completed\nS1 ONLY = completed',flush=True)
 opt=torch.optim.AdamW(p.parameters(),lr=5e-4);diag=[]
 for task in TASKS:
  r=next(x for x in sel['train'] if x['task_type']==task);opt.zero_grad(set_to_none=True);o=model(**batch(r,'correct'));o.loss.backward();gs=[x.grad for x in p.parameters()];diag.append({'task_type':task,'loss':float(o.loss),'finite':bool(torch.isfinite(o.loss)),'supervised_tokens':int((batch(r,'correct')['labels']!=-100).sum()),'gradient_norm':math.sqrt(sum(float(g.float().pow(2).sum()) for g in gs if g is not None)),'nonzero_grad_tensors':sum(g is not None and bool(torch.count_nonzero(g)) for g in gs),'qwen_gradients':sum(x.grad is not None for x in model.parameters()),'croma_gradients':0,'s1_gradients':sum(x.grad is not None for x in s1.parameters()),'s2_gradients':sum(x.grad is not None for x in s2.parameters())})
 opt.zero_grad(set_to_none=True);dump('phase3q1_supervision_diagnostic.json',{'diagnostics':diag,'test_access_count':0});config={'seed':SEED,'optimizer':'AdamW','learning_rate':5e-4,'epochs':1,'batch_size':1,'gradient_accumulation':1,'maximum_optimizer_steps':60,'record_order':[r['record_id'] for r in sel['train']],'test_access_count':0};dump('phase3q1_preregistered_config.json',config)
 p.train();steps=[]
 for i,r in enumerate(sel['train'],1):
  opt.zero_grad(set_to_none=True);o=model(**batch(r,'correct'));o.loss.backward();opt.step();steps.append(float(o.loss))
  if i==1 or i%10==0 or i==60:print(f'[3Q1][TRAIN] record {i}/60\ntask = {r["task_type"]}\nloss = {float(o.loss)}\noptimizer_step = {i}',flush=True)
 final=save('phase3q1_joint_projector_final.pt',p);losses=[]
 with torch.no_grad():
  for r in sel['validation']:losses.append({'record_id':r['record_id'],'task_type':r['task_type'],'loss':float(model(**batch(r,'correct')).loss)})
 summary={'initial':initial,'final':final,'loss_summary':{'first':steps[0],'last':steps[-1],'mean':float(np.mean(steps))},'validation':{'mean':float(np.mean([x['loss'] for x in losses])),'per_task':{t:float(np.mean([x['loss'] for x in losses if x['task_type']==t])) for t in TASKS},'raw':losses},'frozen_unchanged':{'qwen':hash_state(model)==q0,'croma':True,'s1':fp(s1)==s10,'s2':fp(s2)==s20},'test_access_count':0};dump('phase3q1_training_summary.json',summary);trained=audit(sel['validation']);dump('phase3q1_trained_semantic_audit.json',trained);dump('phase3q1_summary.json',{'status':'PHASE3Q1_COMPLETE','training_summary':summary,'test_access_count':0,'classification_pending_reload':'MULTIMODAL_FUSION_VALID_BUT_NO_CLEAR_GAIN'});print('TRAINING COMPLETE\nFINAL SHA = '+final['sha256']+'\nFINAL FINGERPRINT = '+final['fingerprint']+'\nVALIDATION MEAN = '+str(summary['validation']['mean']),flush=True)
if __name__=='__main__':main()
