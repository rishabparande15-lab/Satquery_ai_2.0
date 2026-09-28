"""Frozen bounded Phase 3O.18 MCQ-prior and caption-prefix intervention."""
from __future__ import annotations
import json,math,re,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.phase3o15_contrastive_objective_ablation import CK,FP,SHA,SNAPSHOT,ans,fp,getasset,index,hash_state,S2MultispectralProjector,sha256_file,freeze_qwen,optimizer_parameter_ids,Qwen25VLRGBAdapter,cube,question
from src.eo_vlm.training import build_qwen_token_batch
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.image_conditioned_objective import target_sequence_log_probability
OUT=ROOT/'artifacts/training/phase3o/phase3o18_prior_prefix_intervention'
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def req(x,m):
 if not x:raise RuntimeError('PHASE3O18_FAILED:'+m)
def batch(model,tok,p,row,visual,answer=None):return build_qwen_token_batch(tokenizer=tok,model=model,projector=p,s2=cube(getasset(visual,IX,CACHE)),questions=[question(row)],answers=[answer] if answer else None)
def score(model,tok,p,row,visual,answer):
 b=batch(model,tok,p,row,visual,answer);o=model(**b);return target_sequence_log_probability(o.logits,b['labels']),o.loss,b
def early(logits,labels,n=5):
 pos=(labels[0]!=-100).nonzero().flatten()[:n];return sum(torch.log_softmax(logits[0,i-1].float(),-1)[labels[0,i]] for i in pos)
def generate(model,tok,p,row,visual):
 b=batch(model,tok,p,row,visual);z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=16,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,return_dict_in_generate=True,output_scores=True);ids=z.sequences[0,b['input_ids'].shape[1]:];return tok.decode(ids,skip_special_tokens=True),[int(x) for x in ids]
def parse(text,task):
 m=re.search(r'\b([abcd])\b',text.lower()) if task=='mcq' else None;return m.group(1) if m else (text.strip() or None)
def evaluate_mcq(model,tok,p,rows,mp,lookup):
 out=[]
 with torch.inference_mode():
  for r in rows:
   w=lookup[mp[r['record_id']]];scores={};ranks={}
   for cond,v in [('correct',r),('shuffled',w)]:
    scores[cond]={o['key']:float(score(model,tok,p,r,v,o['key'])[0]) for o in r['options']};ranks[cond]=[k for k,_ in sorted(scores[cond].items(),key=lambda x:x[1],reverse=True)]
   g,_=generate(model,tok,p,r,r);sg,_=generate(model,tok,p,r,w);out.append({'record_id':r['record_id'],'reference':r['answer'],'scores':scores,'rankings':ranks,'generation':g,'shuffled_generation':sg,'parse':parse(g,'mcq'),'shuffled_parse':parse(sg,'mcq')})
 def metrics():
  mar=[x['scores']['correct'][x['reference']]-max(v for k,v in x['scores']['correct'].items() if k!=x['reference']) for x in out];freq=Counter(x['rankings']['correct'][0] for x in out);acc=sum(x['parse']==x['reference'] for x in out);sacc=sum(x['shuffled_parse']==x['reference'] for x in out);return {'target_vs_distractor_margin':sum(mar)/len(mar),'rank1':sum(x['rankings']['correct'][0]==x['reference'] for x in out),'normal_accuracy':acc,'shuffled_accuracy':sacc,'ranking_agreement':sum(x['rankings']['correct']==x['rankings']['shuffled'] for x in out),'option_frequency':freq,'dominant_frequency':max(freq.values()),'parseable':sum(x['parse'] is not None for x in out),'records':out}
 return metrics()
def evaluate_caption(model,tok,p,rows,mp,lookup):
 out=[]
 with torch.inference_mode():
  for r in rows:
   w=lookup[mp[r['record_id']]];cs,_,cb=score(model,tok,p,r,r,ans(r));ss,_,sb=score(model,tok,p,r,w,ans(r));g,ids=generate(model,tok,p,r,r);sg,sids=generate(model,tok,p,r,w);out.append({'record_id':r['record_id'],'generation':g,'shuffled_generation':sg,'token_ids':ids,'shuffled_token_ids':sids,'early_margin':float(early(model(**cb).logits,cb['labels'])-early(model(**sb).logits,sb['labels'])),'reference_margin':float(cs-ss)})
 texts=[x['generation'] for x in out];prefix=lambda n:[tuple(x['token_ids'][:n]) for x in out];return {'early_margin':sum(x['early_margin'] for x in out)/len(out),'token1_unique':len(set(prefix(1))),'prefix3_unique':len(set(prefix(3))),'prefix5_unique':len(set(prefix(5))),'complete_unique':len(set(texts)),'dominant_prefix':Counter(prefix(3)).most_common(1)[0][1],'output_changes':sum(x['generation']!=x['shuffled_generation'] for x in out),'empty':sum(not x['generation'].strip() for x in out),'records':out}
def main():
 pre=json.loads((OUT/'phase3o18_preregistration.json').read_text());req(sha256_file(CK)==SHA,'sha');state=torch.load(CK,map_location='cpu',weights_only=False)['state_dict'];req(fp(state)==FP,'fp');global IX,CACHE;IX=index();CACHE={};runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;req(freeze_qwen(model)['qwen_trainable_parameter_count']==0,'qwen');qh=hash_state(model)
 results={}
 for task in ('mcq','caption'):
  panel=json.loads((OUT/f'phase3o18_{task}_panels.json').read_text());maps=json.loads((OUT/'phase3o18_shuffle_maps.json').read_text())[task];lookup={k:{r['record_id']:r for r in panel[k]} for k in ('train','validation')};mp={k:{x['sample_id']:x['shuffled_sample_id'] for x in maps[k]} for k in ('train','validation')};variants=['base_ce','target_vs_distractor','joint'] if task=='mcq' else ['base_ce','early_token','prefix_weighted'];base=None;vres={}
  for var in variants:
   p=S2MultispectralProjector().to('cuda');p.load_state_dict(state);p.train();req(fp(p.state_dict())==FP,'shared init');ev=evaluate_mcq(model,tok,p,panel['validation'],mp['validation'],lookup['validation']) if task=='mcq' else evaluate_caption(model,tok,p,panel['validation'],mp['validation'],lookup['validation']);base=base or ev;opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=1e-4);req(not optimizer_parameter_ids(opt).intersection(id(x) for x in model.parameters()),'optimizer');steps=[]
   for i,r in enumerate(panel['train'][:8],1):
    print(f'[3O18][{task.upper()}][{var}] step {i}/8',flush=True);w=lookup['train'][mp['train'][r['record_id']]];opt.zero_grad(set_to_none=True);p.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);cs,ce,cb=score(model,tok,p,r,r,ans(r));ss,_,sb=score(model,tok,p,r,w,ans(r));rank=torch.tensor(0.,device='cuda');visual=torch.tensor(0.,device='cuda')
    if task=='mcq':
     ds=[score(model,tok,p,r,r,o['key'])[0] for o in r['options'] if o['key']!=r['answer']];rank=torch.relu(.10-(cs-torch.stack(ds).max()))
     if var=='joint':visual=torch.nn.functional.softplus(-(cs-ss))
     loss=ce+(0 if var=='base_ce' else .25*rank)+(.25*visual if var=='joint' else 0)
    else:
     e1=early(model(**cb).logits,cb['labels']);e2=early(model(**sb).logits,sb['labels']);rank=torch.nn.functional.softplus(-(e1-e2));loss=ce+(0 if var=='base_ce' else (.25 if var=='early_token' else .35)*rank)+(.10*torch.nn.functional.softplus(-(cs-ss)) if var=='prefix_weighted' else 0)
    req(torch.isfinite(loss),'finite');loss.backward();gn=math.sqrt(sum(float(x.grad.detach().float().pow(2).sum()) for x in p.parameters() if x.grad is not None));qg=sum(x.grad is not None for x in model.parameters());req(gn>0 and qg==0,'grads');opt.step();steps.append({'step':i,'base_ce':float(ce),'ranking_or_early':float(rank),'visual':float(visual),'total':float(loss),'grad_norm':gn,'qwen_gradients':qg})
   p.eval();ck=OUT/f'{task}_{var}_projector.pt';torch.save({'state_dict':p.state_dict(),'optimizer_steps':8,'initial_sha':SHA,'initial_fingerprint':FP},ck);ch=sha256_file(ck);fpf=fp(p.state_dict());print(f'POST-TRIAL CHECKPOINT SAVED = {ck.relative_to(ROOT)}\nSHA = {ch}\nFINGERPRINT = {fpf}',flush=True);post=evaluate_mcq(model,tok,p,panel['validation'],mp['validation'],lookup['validation']) if task=='mcq' else evaluate_caption(model,tok,p,panel['validation'],mp['validation'],lookup['validation']);raw=OUT/f'{task}_{var}_raw_validation.json';dump(raw,{'pre':ev['records'],'post':post['records']})
   if task=='mcq':
    gate=post['target_vs_distractor_margin']>ev['target_vs_distractor_margin'] and post['rank1']>ev['rank1'] and post['dominant_frequency']<=ev['dominant_frequency'] and post['normal_accuracy']>=ev['normal_accuracy'] and post['ranking_agreement']<ev['ranking_agreement'] and post['parseable']>=ev['parseable']
   else:gate=post['early_margin']>ev['early_margin'] and (post['token1_unique']>=ev['token1_unique'] or post['prefix3_unique']>=ev['prefix3_unique']) and post['complete_unique']>=ev['complete_unique'] and post['dominant_prefix']<=ev['dominant_prefix'] and post['output_changes']>0 and post['empty']<=ev['empty']
   vres[var]={'pre':{k:v for k,v in ev.items() if k!='records'},'post':{k:v for k,v in post.items() if k!='records'},'steps':steps,'checkpoint':str(ck.relative_to(ROOT)),'sha256':ch,'fingerprint':fpf,'raw':str(raw.relative_to(ROOT)),'gate':'PROMISING' if gate else 'NOT_PROMISING'}
  dump(OUT/f'phase3o18_{task}_results.json',{'task':task,'baseline':{k:v for k,v in base.items() if k!='records'},'variants':vres,'test_access_count':0});results[task]=vres
 req(hash_state(model)==qh and sha256_file(CK)==SHA and fp(torch.load(CK,map_location='cpu',weights_only=False)['state_dict'])==FP,'immutability');dump(OUT/'phase3o18_summary.json',{'status':'PHASE3O18_COMPLETE','qwen_unchanged':True,'phase3o12_unchanged':True,'test_access_count':0,'mcq_best':next((k for k,v in results['mcq'].items() if v['gate']=='PROMISING'),None),'caption_best':next((k for k,v in results['caption'].items() if v['gate']=='PROMISING'),None)});print('PHASE3O18_COMPLETE',flush=True)
if __name__=='__main__':main()
