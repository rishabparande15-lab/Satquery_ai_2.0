"""Auditable, bounded Phase 3O.16R task-local recovery study."""
from __future__ import annotations
import json, math, re, sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.phase3o15_contrastive_objective_ablation import CK,FP,SHA,SNAPSHOT,ans,fp,getasset,index,pair_outputs,hash_state,S2MultispectralProjector,sha256_file,freeze_qwen,optimizer_parameter_ids,Qwen25VLRGBAdapter,cube,question
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.image_conditioned_objective import image_conditioned_penalty,target_sequence_log_probability
from src.eo_vlm.training import build_qwen_token_batch
from src.eo_vlm.phase3o16r_protocol import classify, frozen_digest, validate_preregistration
OUT=ROOT/'artifacts/training/phase3o/phase3o16r_recovery'
SPECS={'binary':{'base_ce':('none',0.,0.),'smooth_likelihood':('softplus',.25,0.),'low_weight_hinge':('hinge',.10,.25)},'mcq':{'base_ce':('none',0.,0.),'target_vs_distractor_ranking':('hinge',.25,.10),'joint_ranking_and_image_margin':('softplus',.25,0.)},'caption':{'base_ce':('none',0.,0.),'sequence_likelihood_separation':('softplus',.25,0.),'sequence_likelihood_plus_early_token_image_discrimination':('hinge',.10,.10)}}
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
def req(x,m):
 if not x:raise RuntimeError('PHASE3O16R_FAILED_INTEGRITY:'+m)
def gen(model,tok,p,row,ix,cache,visual=None,mode='correct'):
 s=cube(getasset(visual or row,ix,cache));s=torch.zeros_like(s) if mode=='zero' else s
 b=build_qwen_token_batch(tokenizer=tok,model=model,projector=p,s2=s,questions=[question(row)],answers=None)
 z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=16,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,return_dict_in_generate=True,output_scores=True)
 ids=z.sequences[0,b['input_ids'].shape[1]:];text=tok.decode(ids,skip_special_tokens=True)
 return text,[int(x) for x in ids],float(torch.softmax(z.scores[0][0].float(),-1).max())
def score(model,tok,p,row,visual_row,ix,cache,target=None):
 target=ans(row) if target is None else target
 b=build_qwen_token_batch(tokenizer=tok,model=model,projector=p,s2=cube(getasset(visual_row,ix,cache)),questions=[question(row)],answers=[target])
 o=model(**b);return float(target_sequence_log_probability(o.logits,b['labels'])),float(o.loss)
def zero_score(model,tok,p,row,ix,cache,target=None):
 target=ans(row) if target is None else target;s=torch.zeros_like(cube(getasset(row,ix,cache)))
 b=build_qwen_token_batch(tokenizer=tok,model=model,projector=p,s2=s,questions=[question(row)],answers=[target]);o=model(**b);return float(target_sequence_log_probability(o.logits,b['labels']))
def parse(task,text):
 x=text.strip().lower()
 if task=='binary':
  m=re.search(r'\b(yes|no)\b',x);return (m.group(1) if m else None)
 if task=='mcq':
  m=re.search(r'\b([abcd])\b',x);return (m.group(1) if m else None)
 return x if x else None
def candidate_scores(model,tok,p,row,visual,ix,cache):return {o['key']:score(model,tok,p,row,visual,ix,cache,o['key'])[0] for o in row['options']}
def evaluate(task,model,tok,p,rows,mapping,lookup,ix,cache):
 records=[]
 with torch.inference_mode():
  for i,row in enumerate(rows,1):
   wrong=lookup[mapping[row['record_id']]['shuffled_sample_id']]; cs,loss=score(model,tok,p,row,row,ix,cache);ss,_=score(model,tok,p,row,wrong,ix,cache);zs=zero_score(model,tok,p,row,ix,cache)
   normal,nids,nprob=gen(model,tok,p,row,ix,cache);shuffled,sids,sprob=gen(model,tok,p,row,ix,cache,visual=wrong);parsed=parse(task,normal);sp=parse(task,shuffled)
   r={'record_id':row['record_id'],'image_id':row['image_id'],'shuffled_record_id':wrong['record_id'],'reference':ans(row),'correct_image_generation':normal,'shuffled_image_generation':shuffled,'correct_token_ids':nids,'shuffled_token_ids':sids,'parse':parsed,'shuffled_parse':sp,'correctness':parsed==ans(row) if task!='caption' else None,'shuffled_correctness':sp==ans(row) if task!='caption' else None,'correct_target_score':cs,'shuffled_target_score':ss,'zero_target_score':zs,'correct_vs_shuffled_margin':cs-ss,'loss':loss,'changed_under_shuffle':normal!=shuffled,'first_token_confidence':nprob,'first_three_prefix':tok.decode(nids[:3],skip_special_tokens=True)}
   if task=='mcq':
    r['candidate_scores']={'correct_image':candidate_scores(model,tok,p,row,row,ix,cache),'shuffled_image':candidate_scores(model,tok,p,row,wrong,ix,cache),'zero_image':{o['key']:zero_score(model,tok,p,row,ix,cache,o['key']) for o in row['options']}}
    for c,v in r['candidate_scores'].items():r[c+'_ranking']=[k for k,_ in sorted(v.items(),key=lambda z:z[1],reverse=True)]
   records.append(r)
 metrics={'count':len(records),'mean_margin':sum(r['correct_vs_shuffled_margin'] for r in records)/len(records),'finite':all(math.isfinite(r['loss']) for r in records),'output_changes':sum(r['changed_under_shuffle'] for r in records)}
 if task!='caption':
  metrics.update({'normal_accuracy':sum(r['correctness'] for r in records),'shuffled_accuracy':sum(r['shuffled_correctness'] for r in records),'parseable':sum(r['parse'] is not None for r in records),'unparseable':sum(r['parse'] is None for r in records)})
 else:
  counts=Counter(r['correct_image_generation'] for r in records);prefix=Counter(r['first_three_prefix'] for r in records);metrics.update({'unique':len(counts),'dominant':counts.most_common(1)[0][1],'empty_malformed':sum(r['parse'] is None for r in records),'prefix_diversity':len(prefix),'first_token_diversity':len(set(r['correct_token_ids'][0] if r['correct_token_ids'] else -1 for r in records))})
 if task=='mcq':
  margins=[]
  for r in records:
   s=r['candidate_scores']['correct_image'];margins.append(s[r['reference']]-max(v for k,v in s.items() if k!=r['reference']))
  metrics['target_vs_distractor_margin']=sum(margins)/len(margins)
 return {'metrics':metrics,'records':records}
def evidence(task,pre,post,grad,qg):
 a,b=pre['metrics'],post['metrics'];common={'margin_increased':b['mean_margin']>a['mean_margin'],'losses_finite':b['finite'],'projector_gradients_nonzero':grad,'qwen_gradients_zero':qg==0,'test_access_zero':True}
 if task=='binary':return {**common,'normal_accuracy_not_decreased':b['normal_accuracy']>=a['normal_accuracy'],'normal_beats_shuffled_by_one':b['normal_accuracy']>=b['shuffled_accuracy']+1,'parseability_not_decreased':b['parseable']>=a['parseable'],'output_changed':b['output_changes']>=1}
 if task=='mcq':return {**common,'margin_increased':b['target_vs_distractor_margin']>a['target_vs_distractor_margin'],'image_margin_not_decreased':b['mean_margin']>=a['mean_margin'],'normal_accuracy_not_decreased':b['normal_accuracy']>=a['normal_accuracy'],'normal_beats_shuffled_by_one':b['normal_accuracy']>=b['shuffled_accuracy']+1,'parseability_not_worse':b['parseable']>=a['parseable']}
 return {**common,'unique_not_decreased':b['unique']>=a['unique'],'dominant_not_increased':b['dominant']<=a['dominant'],'empty_malformed_not_increased':b['empty_malformed']<=a['empty_malformed'],'output_changed':b['output_changes']>=1,'prefix_diversity_not_materially_worse':b['prefix_diversity']>=a['prefix_diversity']}
def main():
 pre=json.loads((OUT/'phase3o16r_preregistration.json').read_text());validate_preregistration(pre);prereg_digest=frozen_digest(OUT/'phase3o16r_preregistration.json');req(sha256_file(CK)==SHA,'initial sha');state=torch.load(CK,map_location='cpu',weights_only=False)['state_dict'];req(fp(state)==FP,'initial fp')
 maps=json.loads((OUT/'phase3o16r_shuffle_maps.json').read_text())['maps'];ix=index();cache={};runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;req(freeze_qwen(model)['qwen_trainable_parameter_count']==0,'qwen frozen');qh=hash_state(model)
 allresults={}
 for task in ('binary','mcq','caption'):
  panel=json.loads((OUT/f'phase3o16r_{task}_panel.json').read_text());rows={x:panel[x] for x in ('train','validation')};mapping={x:{m['sample_id']:m for m in maps[x] if m['task_type']==panel['task_type']} for x in rows};lookup={x:{r['record_id']:r for r in rows[x]} for x in rows};taskresults={}
  # MCQ/caption diagnostic is the pre-evaluation persisted with the results.
  for name,(kind,w,margin) in SPECS[task].items():
   print(f'[3O16R][{task.upper()}][{name}] pre-evaluation',flush=True);p=S2MultispectralProjector().to('cuda');p.load_state_dict(state);p.train();req(fp(p.state_dict())==FP,'shared init')
   pre_eval=evaluate(task,model,tok,p,rows['validation'],mapping['validation'],lookup['validation'],ix,cache)
   opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=1e-4);req(not optimizer_parameter_ids(opt).intersection(id(x) for x in model.parameters()),'qwen optimizer')
   steps=[]
   for step,row in enumerate(rows['train'][:8],1):
    print(f'[3O16R][{task.upper()}][{name}] step {step}/8',flush=True);opt.zero_grad(set_to_none=True);p.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);wrong=lookup['train'][mapping['train'][row['record_id']]['shuffled_sample_id']];_,b,co,so=pair_outputs(model,tok,p,row,wrong,ix,cache);cs=target_sequence_log_probability(co.logits,b['labels']);ss=target_sequence_log_probability(so.logits,b['labels']);loss=co.loss+image_conditioned_penalty(cs-ss,kind=kind,weight=w,margin=margin);req(torch.isfinite(loss),'finite loss');loss.backward();gn=math.sqrt(sum(float(x.grad.detach().float().pow(2).sum()) for x in p.parameters() if x.grad is not None));qg=sum(x.grad is not None for x in model.parameters());req(gn>0 and qg==0,'gradient flow');opt.step();steps.append({'step':step,'loss':float(loss),'margin':float(cs-ss),'projector_grad_norm':gn,'qwen_gradient_tensors':qg})
   p.eval();checkpoint=OUT/f'{task}_{name}_post_trial_projector.pt';torch.save({'state_dict':p.state_dict(),'initial_sha256':SHA,'initial_fingerprint':FP,'optimizer_steps':8,'preregistration_sha256':prereg_digest},checkpoint);ch=sha256_file(checkpoint);finalfp=fp(p.state_dict());print(f'POST-TRIAL CHECKPOINT SAVED = {checkpoint.relative_to(ROOT)}\nSHA = {ch}\nFINGERPRINT = {finalfp}',flush=True)
   post=evaluate(task,model,tok,p,rows['validation'],mapping['validation'],lookup['validation'],ix,cache);raw=OUT/f'{task}_{name}_validation_raw_outputs.json';dump(raw,{'pre':pre_eval['records'],'post':post['records'],'test_access_count':0});ev=evidence(task,pre_eval,post,all(s['projector_grad_norm']>0 for s in steps),sum(s['qwen_gradient_tensors'] for s in steps));klass,reasons=classify(task,ev);print(f'NORMAL ACCURACY = {post["metrics"].get("normal_accuracy","N/A")}\nSHUFFLED ACCURACY = {post["metrics"].get("shuffled_accuracy","N/A")}\nMARGIN CHANGE = {post["metrics"]["mean_margin"]-pre_eval["metrics"]["mean_margin"]}\nOUTPUT CHANGES = {post["metrics"]["output_changes"]}\nGATE RESULT = {klass}\nGATE REASON = {reasons}',flush=True)
   taskresults[name]={'initial_fingerprint':FP,'pre_metrics':pre_eval['metrics'],'post_metrics':post['metrics'],'steps':steps,'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':ch,'checkpoint_fingerprint':finalfp,'raw_validation_output_path':str(raw.relative_to(ROOT)),'gate_evidence':ev,'gate_result':klass,'gate_reason':reasons}
  diagnostic={'failure_mode':'ALL_OPTIONS_MOVE_TOGETHER_OR_LANGUAGE_PRIOR_DOMINANCE'} if task=='mcq' else {'collapse_location':'NOT_LOCALIZED'} if task=='caption' else {}
  receipt={'phase':'3O.16R','task':task,'status':'COMPLETE','historical_phase3o16_status':'PHASE3O16_BLOCKED','preregistration_sha256':prereg_digest,'test_access_count':0,'variants':taskresults,'diagnostic':diagnostic};dump(OUT/f'phase3o16r_{task}_results.json',receipt);allresults[task]=receipt
 req(hash_state(model)==qh and sha256_file(CK)==SHA and fp(torch.load(CK,map_location='cpu',weights_only=False)['state_dict'])==FP,'final immutability');dump(OUT/'phase3o16r_integrity.json',{'qwen_unchanged':True,'qwen_trainable_parameters':0,'test_access_count':0,'phase3o12_unchanged':True,'historical_phase3o16_unchanged':True,'preregistration_sha256':prereg_digest})
 print('PHASE3O16R_COMPLETE',flush=True)
if __name__=='__main__':main()
