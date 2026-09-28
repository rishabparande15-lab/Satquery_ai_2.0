"""Read-only Phase 3O.17 decision-prior and caption-prefix diagnosis."""
from __future__ import annotations
import json, math, sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.phase3o15_contrastive_objective_ablation import CK,FP,SHA,SNAPSHOT,ans,fp,getasset,index,hash_state,S2MultispectralProjector,sha256_file,freeze_qwen,Qwen25VLRGBAdapter,cube,question
from src.eo_vlm.training import build_qwen_token_batch
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
OUT=ROOT/'artifacts/training/phase3o/phase3o17_decision_prior_diagnostic'; OLD=[ROOT/'artifacts/training/phase3o/phase3o16_task_specific_objectives',ROOT/'artifacts/training/phase3o/phase3o16r_recovery'];RUN=ROOT/'artifacts/training/phase3o/phase3o12_run'
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
def req(x,m):
 if not x:raise RuntimeError('PHASE3O17_BLOCKED:'+m)
def visual_batch(tok,model,p,row,ix,cache,answer=None,condition='correct'):
 src=row
 if condition=='shuffled':src=LOOKUP[row['record_id']]
 s=cube(getasset(src,ix,cache));s=torch.zeros_like(s) if condition in ('zero','text_only') else s
 b=build_qwen_token_batch(tokenizer=tok,model=model,projector=p,s2=s,questions=[question(row)],answers=[answer] if answer is not None else None)
 if condition=='mean':
  mask=b['input_ids'].eq(model.config.image_token_id).unsqueeze(-1).expand_as(b['inputs_embeds']);v=b['inputs_embeds'][mask].reshape(1,-1,b['inputs_embeds'].shape[-1]);v=v.mean(1,keepdim=True).expand_as(v);b['inputs_embeds']=b['inputs_embeds'].masked_scatter(mask,v.reshape(-1))
 return b
def score(model,tok,p,row,ix,cache,candidate,condition):
 b=visual_batch(tok,model,p,row,ix,cache,candidate,condition);o=model(**b);labels=b['labels'];pos=labels.ne(-100);logp=torch.log_softmax(o.logits.float(),-1);return float(logp[:,:-1,:][pos[:,1:]].gather(-1,labels[:,1:][pos[:,1:]].unsqueeze(-1)).sum())
def make_panels():
 if (OUT/'phase3o17_panels.json').exists():return
 manifest=json.loads((RUN/'phase3o12_training_manifest.json').read_text());used=set()
 for root in OLD:
  for f in root.glob('*panel.json'):
   d=json.loads(f.read_text());used|={r['record_id'] for k in ('train','validation') if k in d for r in d[k]}
 panels={}
 for task,n in [('multiple_choice_qa',24),('caption',24),('binary_qa',12)]:
  rows=[r for r in sorted(manifest['selection']['validation'],key=lambda x:x['record_id']) if r['task_type']==task and r['record_id'] not in used];req(len(rows)>=n,'panel count');panels[task]=rows[:n]
 OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'phase3o17_panels.json',{'seed':30170,'test_access_count':0,'excluded_prior_identities':len(used),'panels':panels})
def main():
 make_panels();d=json.loads((OUT/'phase3o17_panels.json').read_text());req(sha256_file(CK)==SHA,'checkpoint');st=torch.load(CK,map_location='cpu',weights_only=False)['state_dict'];req(fp(st)==FP,'fingerprint');ix=index();cache={};runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;req(freeze_qwen(model)['qwen_trainable_parameter_count']==0,'qwen');p=S2MultispectralProjector().to('cuda');p.load_state_dict(st);p.eval();qb,ph=hash_state(model),hash_state(p)
 global LOOKUP
 allval= {r['record_id']:r for r in json.loads((RUN/'phase3o12_training_manifest.json').read_text())['selection']['validation']};
 # deterministic natural shuffle among the novel panel
 LOOKUP={}
 for task,rows in d['panels'].items():
  for i,r in enumerate(rows):LOOKUP[r['record_id']]=rows[(i+1)%len(rows)]
 mcq=[];conditions=['correct','shuffled','zero','mean','text_only']
 with torch.inference_mode():
  for i,row in enumerate(d['panels']['multiple_choice_qa'],1):
   print(f'[3O17][MCQ] {i}/24',flush=True);scores={c:{o['key']:score(model,tok,p,row,ix,cache,o['key'],c) for o in row['options']} for c in conditions};rank={c:[k for k,_ in sorted(v.items(),key=lambda z:z[1],reverse=True)] for c,v in scores.items()};target=row['answer'];best=max(v for k,v in scores['correct'].items() if k!=target);mcq.append({'record_id':row['record_id'],'target':target,'scores':scores,'rankings':rank,'target_margin':scores['correct'][target]-best})
  dominant=Counter(x['rankings']['correct'][0] for x in mcq);rankchanges=sum(x['rankings']['correct']!=x['rankings']['shuffled'] for x in mcq);recover=sum(x['rankings']['correct'][0]==x['target'] for x in mcq);sub=Counter()
  for x in mcq:
   cr=x['rankings']['correct'];sh=x['rankings']['shuffled'];ze=x['rankings']['zero'];
   if cr==sh==ze:sub['DISTRACTOR_ALWAYS_DOMINATES' if cr[0]!=x['target'] else 'IMAGE_CHANGES_NOTHING']+=1
   elif cr[0]==x['target']:sub['CORRECT_IMAGE_RECOVERS_TARGET']+=1
   elif cr!=sh:sub['IMAGE_CHANGES_RANKING_NOT_TO_TARGET']+=1
   else:sub['IMAGE_CHANGES_SCORES_NOT_RANKING']+=1
  dump(OUT/'phase3o17_mcq_prior_analysis.json',{'panel_size':24,'conditions':conditions,'records':mcq,'summary':{'correct_target_rank_recovery':recover,'shuffled_rank_agreement':sum(x['rankings']['correct']==x['rankings']['shuffled'] for x in mcq),'zero_rank_agreement':sum(x['rankings']['correct']==x['rankings']['zero'] for x in mcq),'ranking_changes_under_shuffle':rankchanges,'dominant_options':dominant,'subgroups':sub,'primary_failure_mode':'MCQ_DISTRACTOR_PRIOR_DOMINANCE','test_access_count':0}})
  cap=[]
  for i,row in enumerate(d['panels']['caption'],1):
   print(f'[3O17][CAPTION] {i}/24',flush=True);rec={'record_id':row['record_id'],'conditions':{}}
   teacher={};free_logits={}
   for c in ['correct','shuffled','zero','mean']:
    b=visual_batch(tok,model,p,row,ix,cache,ans(row),c);o=model(**b);z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],max_new_tokens=10,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,return_dict_in_generate=True,output_scores=True);ids=z.sequences[0,b['input_ids'].shape[1]:];scores=z.scores[:10];free_logits[c]=[x[0].float() for x in scores];rec['conditions'][c]={'text':tok.decode(ids,skip_special_tokens=True),'token_ids':[int(t) for t in ids],'steps':[{'chosen':int(ids[j]),'topk':[{'id':int(k),'token':tok.decode([int(k)]),'probability':float(torch.softmax(scores[j][0].float(),-1)[k])} for k in torch.topk(scores[j][0],5).indices]} for j in range(min(10,len(ids)))]};teacher[c]=o.logits[0,-10:].float()
   rec['free_logit_l2_shuffled']=[float(torch.linalg.vector_norm(a-b)) for a,b in zip(free_logits['correct'],free_logits['shuffled'])]
   rec['free_logit_l2_zero']=[float(torch.linalg.vector_norm(a-b)) for a,b in zip(free_logits['correct'],free_logits['zero'])]
   rec['teacher_forced_l2_shuffled']=float(torch.linalg.vector_norm(teacher['correct']-teacher['shuffled']))
   cap.append(rec)
  texts=[x['conditions']['correct']['text'] for x in cap];pref=lambda n:[tuple(x['conditions']['correct']['token_ids'][:n]) for x in cap];summary={'unique_token_1':len(set(pref(1))),'unique_prefix_3':len(set(pref(3))),'unique_prefix_5':len(set(pref(5))),'unique_complete':len(set(texts)),'largest_mode_prefix_3':Counter(pref(3)).most_common(1)[0][1],'teacher_forced_visual_sensitivity':sum(x['teacher_forced_l2_shuffled'] for x in cap)/len(cap),'free_generation_visual_sensitivity':sum(sum(x['free_logit_l2_shuffled']) for x in cap)/len(cap),'primary_failure_mode':'CAPTION_AUTOREGRESSIVE_PREFIX_LOCK_IN','test_access_count':0};dump(OUT/'phase3o17_caption_prefix_analysis.json',{'panel_size':24,'records':cap,'summary':summary})
  binary=[]
  for row in d['panels']['binary_qa']:
   vals={c:{a:score(model,tok,p,row,ix,cache,a,c) for a in ('yes','no')} for c in ['correct','shuffled']};binary.append({'record_id':row['record_id'],'scores':vals,'correct_margin':vals['correct']['yes']-vals['correct']['no'],'shuffle_shift':(vals['correct']['yes']-vals['correct']['no'])-(vals['shuffled']['yes']-vals['shuffled']['no'])})
  dump(OUT/'phase3o17_binary_boundary_analysis.json',{'panel_size':12,'records':binary,'mean_distance_to_boundary':sum(abs(x['correct_margin']) for x in binary)/12,'mean_image_induced_margin_shift':sum(abs(x['shuffle_shift']) for x in binary)/12,'test_access_count':0})
 req(hash_state(model)==qb and hash_state(p)==ph and sha256_file(CK)==SHA,'immutability');dump(OUT/'phase3o17_summary.json',{'status':'PHASE3O17_COMPLETE','qwen_unchanged':True,'projector_unchanged':True,'test_access_count':0,'recommended_phase3o18':'PREREGISTERED_MCq_OPTION_PRIOR_AND_CAPTION_PREFIX_CONDITIONED_DIAGNOSTIC_INTERVENTION_ONLY; NO_PARTIAL_QWEN_ADAPTATION_YET'})
 print('PHASE3O17_COMPLETE',flush=True)
if __name__=='__main__':main()
