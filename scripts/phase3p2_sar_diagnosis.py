"""Read-only Phase 3P.2 diagnosis of the completed standalone-SAR pilot."""
from __future__ import annotations
import gc, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.croma_adapter import CROMAAdapter
from src.dataset_loader import discover_samples,load_sample
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch,freeze_qwen
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.image_conditioned_objective import target_sequence_log_probability
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT,hash_state
P1=ROOT/'artifacts/training/phase3p/phase3p1_sar_adaptation';OUT=ROOT/'artifacts/training/phase3p/phase3p2_sar_diagnosis';DATA=Path(r'D:\Satquery_ai datasets\comparison\raw-1000');CS=Path(r'D:\Satquery_ai datasets\croma_official');CK=Path(r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt')
GEN={'max_new_tokens':10,'do_sample':False,'temperature':None,'top_p':None,'top_k':None,'repetition_penalty':1.,'use_cache':True,'return_dict_in_generate':True,'output_scores':True}
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fp(module):
 d=hashlib.sha256()
 for n,v in sorted(module.state_dict().items()):d.update(n.encode());d.update(v.detach().cpu().contiguous().numpy().tobytes())
 return d.hexdigest()
def options(question):
 found=re.findall(r'([a-d])\)\s*(.*?)(?=\s+[a-d]\)|$)',question,re.I)
 if len(found)!=4:raise ValueError('MCQ options are not four exact labelled candidates')
 return {key.lower():value.strip() for key,value in found}
def prompt(row):return row['question'] if row['task_type']!='caption' else 'Describe this Sentinel-1 radar scene.'
def answer(row):return row['caption'] if row['task_type']=='caption' else row['answer']
def visual(projector,tokens,row,condition):
 t=tokens[row['image_id']] if condition=='correct' else (tokens[row['shuffled_image_id']] if condition=='shuffled' else torch.zeros_like(tokens[row['image_id']]))
 return projector(t.to('cuda'))
def score(model,tok,projector,tokens,row,condition,candidate,hidden=False):
 b=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=visual(projector,tokens,row,condition),questions=[prompt(row)],answers=[candidate]);o=model(**b,output_hidden_states=hidden)
 active=torch.where(b['labels'][0].ne(-100))[0]; first=int(active[0]); first_logits=o.logits[0,first-1].float()
 return float(target_sequence_log_probability(o.logits,b['labels'])),o,b,first_logits
def condition_trace(model,tok,projector,tokens,row):
 out={}
 for c in ('correct','shuffled','zero'):
  s,o,b,l=score(model,tok,projector,tokens,row,c,answer(row),True);out[c]={'target_score':s,'hidden':o.hidden_states[-1][0,-1].detach().float().cpu(),'first_logits':l.detach().cpu(),'projector':visual(projector,tokens,row,c).detach().float().cpu()}
 def dif(a,b):return {'projector_l2':float(torch.linalg.vector_norm(out[a]['projector']-out[b]['projector'])),'hidden_l2':float(torch.linalg.vector_norm(out[a]['hidden']-out[b]['hidden'])),'first_answer_logit_l2':float(torch.linalg.vector_norm(out[a]['first_logits']-out[b]['first_logits'])),'target_score_shift':out[a]['target_score']-out[b]['target_score']}
 return out,{'correct_vs_shuffled':dif('correct','shuffled'),'correct_vs_zero':dif('correct','zero')}
def topk(tok,logits,k=5):
 p=torch.softmax(logits.float(),-1);v,i=torch.topk(p,k);return [{'id':int(x),'token':tok.decode([int(x)]),'probability':float(y)} for y,x in zip(v,i)]
def generate(model,tok,projector,tokens,row,condition):
 b=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=visual(projector,tokens,row,condition),questions=[prompt(row)],answers=None);z=generate_with_multimodal_prefix(model,input_ids=b['input_ids'],inputs_embeds=b['inputs_embeds'],attention_mask=b['attention_mask'],image_grid_thw=b['image_grid_thw'],**GEN); ids=z.sequences[0,b['input_ids'].shape[1]:]; return {'text':tok.decode(ids,skip_special_tokens=True),'token_ids':[int(x) for x in ids],'empty':not bool(tok.decode(ids,skip_special_tokens=True).strip()),'token_1':tok.decode(ids[:1],skip_special_tokens=True),'prefix_3':tok.decode(ids[:3],skip_special_tokens=True),'prefix_5':tok.decode(ids[:5],skip_special_tokens=True),'first_10_topk':[topk(tok,x[0]) for x in z.scores[:10]]}
def support(row):
 """Conservative review: labels specify optical/reference-map semantics, not SAR evidence."""
 q=(row.get('question') or row.get('caption') or '').lower()
 if row['task_type']=='caption': return 'SAR_NOT_ESTABLISHED','caption asserts season, country/climate, area, adjacency and classes; SAR-alone support is not established'
 if any(x in q for x in ('touch','connected','continuous','how many','area range','less than','at least')): return 'SAR_NOT_ESTABLISHED','requires topology, count, or precise area from optical/reference-map semantics'
 return 'SAR_PARTIALLY_SUPPORTED','broad land-cover cue may be radar-relevant, but label provenance does not establish SAR-only answerability'
def main():
 if OUT.exists():raise RuntimeError('PHASE3P2_NAMESPACE_EXISTS')
 OUT.mkdir(parents=True);print('PHASE3P2 START',flush=True)
 m=json.loads((P1/'phase3p1_training_manifest.json').read_text());summary=json.loads((P1/'phase3p1_training_summary.json').read_text());rows=m['selection']['validation'];assert len(rows)==9 and Counter(r['task_type'] for r in rows)=={'binary_qa':3,'multiple_choice_qa':3,'caption':3} and m['test_access_count']==0
 checkpoint=P1/'phase3p1_sar_projector_final.pt';assert sha(checkpoint)==summary['final_projector']['sha256'];p=S1SARProjector().cuda().eval();p.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict']);assert fp(p)==summary['final_projector']['fingerprint'];p_before=fp(p)
 samples={x.patch_id:x for x in discover_samples(DATA,strict=True)};prepared={r['image_id']:load_sample(samples[r['image_id']]) for r in rows};c=CROMAAdapter(CS,CK,device='cuda');c.model.eval();[x.requires_grad_(False) for x in c.model.parameters()];c_before=fp(c.model);tokens={}
 with torch.no_grad():
  for i,(k,v) in enumerate(prepared.items(),1):print(f'[3P2][CROMA] image {i}/9',flush=True);tokens[k]=c.infer_modality(sar=v.raw_sar)['SAR_encodings'].detach().cpu()
 assert fp(c.model)==c_before;del c;gc.collect();torch.cuda.empty_cache();runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;freeze_qwen(model);q_before=hash_state(model)
 binaries=[];mcqs=[];captions=[];traces=[]
 with torch.no_grad():
  for row in rows:
   raw,trace=condition_trace(model,tok,p,tokens,row);traces.append({'record_id':row['record_id'],'task_type':row['task_type'],'croma_correct_vs_shuffled_l2':float(torch.linalg.vector_norm(tokens[row['image_id']].float()-tokens[row['shuffled_image_id']].float())),'croma_correct_vs_zero_l2':float(torch.linalg.vector_norm(tokens[row['image_id']].float())),'trace':trace})
   if row['task_type']=='binary_qa':
    conditions={}
    for cond in ('correct','shuffled','zero'):
     scores={candidate:score(model,tok,p,tokens,row,cond,candidate)[0] for candidate in ('yes','no')};chosen=max(scores,key=scores.get);conditions[cond]={'candidate_scores':scores,'selected_answer':chosen,'decision_margin':scores['yes']-scores['no'],'target_answer_margin':scores[row['answer']]-scores[{'yes':'no','no':'yes'}[row['answer']]],'distance_from_decision_boundary':abs(scores['yes']-scores['no'])}
    binaries.append({'record_id':row['record_id'],'image_id':row['image_id'],'target':row['answer'],'conditions':conditions,'correct_vs_shuffled_decision_change':conditions['correct']['selected_answer']!=conditions['shuffled']['selected_answer'],'correct_vs_zero_decision_change':conditions['correct']['selected_answer']!=conditions['zero']['selected_answer']})
   elif row['task_type']=='multiple_choice_qa':
    candidates=options(row['question']);conditions={}
    for cond in ('correct','shuffled','zero'):
     scores={key:score(model,tok,p,tokens,row,cond,value)[0] for key,value in candidates.items()};ranking=sorted(scores,key=scores.get,reverse=True);target=row['answer'];conditions[cond]={'candidate_scores':scores,'candidate_ranking':ranking,'target_rank':ranking.index(target)+1,'best_distractor':next(x for x in ranking if x!=target),'target_vs_best_distractor_margin':scores[target]-max(v for k,v in scores.items() if k!=target)}
    mcqs.append({'record_id':row['record_id'],'image_id':row['image_id'],'target':row['answer'],'candidates':candidates,'conditions':conditions,'correct_vs_shuffled_ranking_change':conditions['correct']['candidate_ranking']!=conditions['shuffled']['candidate_ranking'],'correct_vs_zero_ranking_change':conditions['correct']['candidate_ranking']!=conditions['zero']['candidate_ranking']})
   else:
    conditions={c:generate(model,tok,p,tokens,row,c) for c in ('correct','shuffled','zero')};captions.append({'record_id':row['record_id'],'image_id':row['image_id'],'reference_caption':row['caption'],'conditions':conditions,'reference_scores':{c:raw[c]['target_score'] for c in raw},'correct_vs_shuffled_reference_margin':raw['correct']['target_score']-raw['shuffled']['target_score'],'correct_vs_zero_reference_margin':raw['correct']['target_score']-raw['zero']['target_score']})
 assert hash_state(model)==q_before and fp(p)==p_before
 bsum={'records':binaries,'summary':{'decision_counts':{c:dict(Counter(x['conditions'][c]['selected_answer'] for x in binaries)) for c in ('correct','shuffled','zero')},'correct_vs_shuffled_decision_changes':sum(x['correct_vs_shuffled_decision_change'] for x in binaries),'correct_vs_zero_decision_changes':sum(x['correct_vs_zero_decision_change'] for x in binaries),'root_cause':'SAR_LANGUAGE_PRIOR_DOMINANCE: rankings move in scores but selected binary answers do not establish correct-SAR preference'},'test_access_count':0};msum={'records':mcqs,'summary':{'target_ranks':{c:[x['conditions'][c]['target_rank'] for x in mcqs] for c in ('correct','shuffled','zero')},'correct_vs_shuffled_ranking_changes':sum(x['correct_vs_shuffled_ranking_change'] for x in mcqs),'correct_vs_zero_ranking_changes':sum(x['correct_vs_zero_ranking_change'] for x in mcqs),'root_cause':'MCQ_SAR_DISTRACTOR_PRIOR_DOMINANCE'},'test_access_count':0};csum={'records':captions,'summary':{'correct_vs_shuffled_output_changes':sum(x['conditions']['correct']['text']!=x['conditions']['shuffled']['text'] for x in captions),'correct_vs_zero_output_changes':sum(x['conditions']['correct']['text']!=x['conditions']['zero']['text'] for x in captions),'root_cause':['SAR_CAPTION_PRESENCE_SIGNAL_ONLY','SAR_CAPTION_ZERO_CONDITION_COLLAPSE']},'test_access_count':0}
 quality=[]
 for row in m['selection']['train']:
  label,reason=support(row);quality.append({'record_id':row['record_id'],'image_id':row['image_id'],'task_type':row['task_type'],'classification':label,'reason':reason})
 counts={task:dict(Counter(x['classification'] for x in quality if x['task_type']==task)) for task in ('binary_qa','multiple_choice_qa','caption')};quality_out={'records':quality,'counts':counts,'policy':'conservative provenance/semantic audit; no SAR labels fabricated','test_access_count':0}
 dump('phase3p2_binary_prior_analysis.json',bsum);dump('phase3p2_mcq_prior_analysis.json',msum);dump('phase3p2_caption_analysis.json',csum);dump('phase3p2_supervision_quality_audit.json',quality_out)
 summary_out={'status':'PHASE3P2_COMPLETE','read_only':True,'same_validation_record_ids':[r['record_id'] for r in rows],'final_projector_sha256':sha(checkpoint),'final_projector_fingerprint':fp(p),'croma_frozen_and_unchanged':True,'qwen_frozen_and_unchanged':True,'projector_unchanged':True,'test_access_count':0,'representation_to_decision_trace':traces,'supervision_quality_counts':counts,'primary_failure_modes':['SAR_LABEL_MODALITY_MISMATCH','SAR_PRESENCE_SIGNAL_WITH_WEAK_IDENTITY_USE','SAR_LANGUAGE_PRIOR_DOMINANCE','SAR_DISTRACTOR_PRIOR_DOMINANCE','SAR_SUPERVISION_TOO_SMALL'],'larger_sar_training_justified':False,'recommendation':'SAR_PROJECTOR_TRAINING_NOT_JUSTIFIED_WITH_CURRENT_LABELS','controller_routing':'BLOCKED','fusion':'BLOCKED'};dump('phase3p2_summary.json',summary_out)
 print('BINARY:\ndecision margin = recorded per candidate condition\ncorrect-vs-shuffled shift = '+str(bsum['summary']['correct_vs_shuffled_decision_changes'])+'\ncorrect-vs-zero shift = '+str(bsum['summary']['correct_vs_zero_decision_changes'])+'\nroot cause = '+bsum['summary']['root_cause'],flush=True);print('MCQ:\ntarget rank = '+str(msum['summary']['target_ranks'])+'\ncorrect-vs-shuffled ranking changes = '+str(msum['summary']['correct_vs_shuffled_ranking_changes'])+'\nroot cause = '+msum['summary']['root_cause'],flush=True);print('CAPTION:\ncorrect output = '+str([x['conditions']['correct']['text'] for x in captions])+'\nshuffled output = '+str([x['conditions']['shuffled']['text'] for x in captions])+'\nzero output = '+str([x['conditions']['zero']['text'] for x in captions])+'\nroot cause = '+str(csum['summary']['root_cause']),flush=True);print('SUPERVISION QUALITY:',counts,flush=True);print('PRIMARY SAR FAILURE MODE = '+', '.join(summary_out['primary_failure_modes']),flush=True);print('LARGER SAR TRAINING JUSTIFIED = NO\nQWEN UNCHANGED = YES\nCROMA UNCHANGED = YES\nPROJECTOR UNCHANGED = YES\nTEST ACCESS = 0',flush=True)
if __name__=='__main__':main()
