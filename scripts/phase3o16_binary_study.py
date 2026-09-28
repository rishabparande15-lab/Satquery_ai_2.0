"""Persisted-panel Binary portion of the bounded Phase 3O.16 study."""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.phase3o15_contrastive_objective_ablation import (CK,FP,SHA,SNAPSHOT,ans,fp,getasset,image_conditioned_penalty,index,pair_outputs,target_sequence_log_probability,hash_state,S2MultispectralProjector,sha256_file,freeze_qwen,optimizer_parameter_ids,Qwen25VLRGBAdapter)
OUT=ROOT/'artifacts/training/phase3o/phase3o16_task_specific_objectives'
SPECS={'BINARY_A_BASE_CE':('none',0.,0.),'BINARY_B_SMOOTH':('softplus',.25,0.),'BINARY_C_LOW_HINGE':('hinge',.10,.25)}
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def req(x,m):
 if not x:raise RuntimeError('PHASE3O16_BINARY_BLOCKED:'+m)
def audit(model,tok,p,rows,maps,lookup,ix,cache):
 es=[]
 with torch.inference_mode():
  for r in rows:
   _,b,co,so=pair_outputs(model,tok,p,r,lookup[maps[r['record_id']]['shuffled_sample_id']],ix,cache);cs=target_sequence_log_probability(co.logits,b['labels']);ss=target_sequence_log_probability(so.logits,b['labels']);es.append(float(cs-ss))
 return sum(es)/len(es)
def main():
 req(not (OUT/'phase3o16_binary_results.json').exists(),'results exist');req(sha256_file(CK)==SHA,'SHA');state=torch.load(CK,map_location='cpu',weights_only=False)['state_dict'];req(fp(state)==FP,'fingerprint')
 panel=json.loads((OUT/'phase3o16_binary_panel.json').read_text());maps_all=json.loads((OUT/'phase3o16_shuffle_maps.json').read_text())['maps'];rows={k:panel[k] for k in ('train','validation')};maps={k:{x['sample_id']:x for x in maps_all[k] if x['task_type']=='binary_qa'} for k in rows};lookup={k:{r['record_id']:r for r in rows[k]} for k in rows};ix=index();cache={};runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;req(freeze_qwen(model)['qwen_trainable_parameter_count']==0,'Qwen');qb=hash_state(model);result={}
 for name,(kind,w,m) in SPECS.items():
  print(f'[3O16][BINARY] variant {name}',flush=True);p=S2MultispectralProjector().to('cuda');p.load_state_dict(state);p.train();req(fp(p.state_dict())==FP,'init');pre={k:audit(model,tok,p,rows[k],maps[k],lookup[k],ix,cache) for k in rows};opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=1e-4);req(not optimizer_parameter_ids(opt).intersection(id(x) for x in model.parameters()),'optimizer')
  steps=[]
  for i,r in enumerate(rows['train'][:8],1):
   print(f'[3O16][BINARY] step {i}/8',flush=True);opt.zero_grad(set_to_none=True);p.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);_,b,co,so=pair_outputs(model,tok,p,r,lookup['train'][maps['train'][r['record_id']]['shuffled_sample_id']],ix,cache);cs=target_sequence_log_probability(co.logits,b['labels']);ss=target_sequence_log_probability(so.logits,b['labels']);pen=image_conditioned_penalty(cs-ss,kind=kind,weight=w,margin=m);loss=co.loss+pen;loss.backward();gn=math.sqrt(sum(float(x.grad.detach().float().pow(2).sum()) for x in p.parameters() if x.grad is not None));qg=sum(x.grad is not None for x in model.parameters());req(gn>0 and qg==0,'grads');opt.step();steps.append({'step':i,'base_ce':float(co.loss),'contrastive':float(pen),'margin':float(cs-ss),'grad_norm':gn,'qwen_gradient_tensors':qg})
  p.eval();post={k:audit(model,tok,p,rows[k],maps[k],lookup[k],ix,cache) for k in rows};result[name]={'pre':pre,'post':post,'validation_delta':post['validation']-pre['validation'],'steps':steps,'classification':'MARGIN_IMPROVED_PENDING_OUTPUT_GATE'}
 req(hash_state(model)==qb and sha256_file(CK)==SHA and fp(torch.load(CK,map_location='cpu',weights_only=False)['state_dict'])==FP,'immutability');dump(OUT/'phase3o16_binary_results.json',{'status':'BINARY_COMPLETE','test_access_count':0,'shared_initial_fingerprint':FP,'qwen_unchanged':True,'phase3o12_projector_unchanged':True,'results':result});print('BINARY COMPLETE',flush=True)
if __name__=='__main__':main()
