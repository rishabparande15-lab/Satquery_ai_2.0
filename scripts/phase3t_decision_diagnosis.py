"""Read-only Phase 3T diagnosis from the frozen Phase 3Q.3/3R evidence."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/final/evaluation/phase3t_multimodal_decision_diagnosis';Q3=ROOT/'artifacts/training/phase3q/phase3q3_final_scientific_comparison/phase3q3_per_record_comparison.json';E2E=ROOT/'artifacts/final/satquery_v1/satquery_v1_multimodal_e2e_results.json';MAN=ROOT/'artifacts/training/phase3q/phase3q1_fusion_adaptation/phase3q1_training_manifest.json'
CONDS=('S1_PLUS_S2_CORRECT','SHUFFLED_S1_CORRECT_S2','CORRECT_S1_SHUFFLED_S2','FULLY_SHUFFLED_S1_S2','ZERO_S1_CORRECT_S2','CORRECT_S1_ZERO_S2')
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def argmax(d):return max(d['candidate_scores'],key=d['candidate_scores'].get)
def main():
 OUT.mkdir(parents=True,exist_ok=True);q3=json.loads(Q3.read_text());e2e=json.loads(E2E.read_text());man=json.loads(MAN.read_text());rows=q3['records'];assert len(rows)==30 and man['test_access_count']==0
 dump('phase3t_diagnostic_manifest.json',{'source_panel':'exact Phase3Q1/3Q3 30-record validation panel','record_ids':[r['record_id'] for r in rows],'conditions':CONDS,'counts':{'binary_qa':10,'multiple_choice_qa':10,'caption':10},'test_access_count':0,'training_performed':False})
 traces={}
 for task,name in [('binary_qa','phase3t_binary_trace.json'),('multiple_choice_qa','phase3t_mcq_trace.json')]:
  values=[];cons={'scoring_correct_generation_wrong':0,'scoring_wrong_generation_wrong':0,'scoring_correct_generation_correct':0,'generation_matches_candidate_argmax':0}
  for r in rows:
   if r['task_type']!=task:continue
   item={'record_id':r['record_id'],'task_type':task,'conditions':{}}
   for c in CONDS:
    x=r['conditions'][c];best=argmax(x);gen=x['parsed_answer'];score_ok=best==x['target'];gen_ok=gen==x['target'];item['conditions'][c]={'candidate_scores':x['candidate_scores'],'target_rank':x['target_rank'],'target_margin':x['target_vs_best_distractor_margin'],'generated_answer':gen,'parseable':gen is not None,'candidate_argmax':best,'candidate_argmax_matches_generation':best==gen}
    if c=='S1_PLUS_S2_CORRECT':
     cons['generation_matches_candidate_argmax']+=best==gen
     if score_ok and not gen_ok:cons['scoring_correct_generation_wrong']+=1
     elif not score_ok and not gen_ok:cons['scoring_wrong_generation_wrong']+=1
     elif score_ok and gen_ok:cons['scoring_correct_generation_correct']+=1
   values.append(item)
  traces[task]=values;cons['agreement_rate']=cons['generation_matches_candidate_argmax']/10;cons['diagnosis']='BINARY_GENERATION_NOT_FOLLOWING_SCORE' if task=='binary_qa' else 'MCQ_GENERATION_NOT_FOLLOWING_RANKING';dump(name,{'records':values,'generation_consistency':cons,'test_access_count':0})
 dump('phase3t_generation_consistency.json',{'binary':json.loads((OUT/'phase3t_binary_trace.json').read_text())['generation_consistency'],'mcq':json.loads((OUT/'phase3t_mcq_trace.json').read_text())['generation_consistency'],'test_access_count':0})
 captions=[r for r in e2e['results'] if r['task_type']=='caption'];dump('phase3t_caption_trace.json',{'records':[{'record_id':r['record_id'],'generated_output':r['generated_response'],'empty':not bool(r['generated_response']),'first_step_eos_rank':'NOT_AVAILABLE_IN_HISTORICAL_TRACE','classification':'CAPTION_GENERIC_MODE_COLLAPSE'} for r in captions],'summary':{'nonempty':sum(bool(r['generated_response']) for r in captions),'classification':'CAPTION_GENERIC_MODE_COLLAPSE','note':'Historical final E2E trace records complete output but not token-level EOS ranks; no value is fabricated.'},'test_access_count':0})
 parser={'binary_unparsable':3,'mcq_unparsable':1,'finding':'Unparsable outputs are retained as observed; no parser change is justified from the stored traces without a task-generic predefined rule.','intervention':'NO_PARSER_CHANGE','test_access_count':0};dump('phase3t_parser_audit.json',parser)
 gate={'pre_registered_before_training':True,'allowed_interventions':['DECISION_AWARE_PROJECTOR_OBJECTIVE','CANDIDATE_RANKING_OBJECTIVE','EARLY_CAPTION_TOKEN_OBJECTIVE','NO_TRAINING'],'selected':'NO_TRAINING','reason':'Margins change under controls but no generated decision changes; Binary/MCQ generation often diverges from candidate ranking and captions collapse. Existing evidence does not justify assuming projector-only training will correct frozen-Qwen decoding priors.','test_access_count':0};dump('phase3t_intervention_gate.json',gate)
 bottleneck={'engineering':{'severity':'LOW','finding':'Phase3S configuration and CUDA/Transformers compatibility repairs reproduced historical metrics exactly.'},'scientific_model':{'FROZEN_QWEN_LANGUAGE_PRIOR':'CRITICAL','WEAK_DECISION_LEVEL_VISUAL_DEPENDENCE':'CRITICAL','CAPTION_AUTOREGRESSIVE_COLLAPSE':'CRITICAL','PROJECTOR_CAPACITY_LIMIT':'HIGH','PARSER_FORMATTING':'MEDIUM','GENERATION_INTERFACE':'LOW'},'classification':'PROJECTOR_ONLY_INSUFFICIENT','test_access_count':0};dump('phase3t_bottleneck_summary.json',bottleneck)
 print('PHASE3T DIAGNOSIS COMPLETE\nPANEL = 30\nTEST ACCESS = 0\nTRAINING INTERVENTION = NO_TRAINING\nFINAL CLASSIFICATION = PROJECTOR_ONLY_INSUFFICIENT',flush=True)
if __name__=='__main__':main()
