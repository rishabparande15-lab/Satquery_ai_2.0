"""Prepare immutable Phase 3O.16R panels and machine-readable gates."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eo_vlm.phase3o16r_protocol import canonical_sha, validate_preregistration
RUN=ROOT/'artifacts/training/phase3o/phase3o12_run'; OLD=ROOT/'artifacts/training/phase3o/phase3o16_task_specific_objectives'; OUT=ROOT/'artifacts/training/phase3o/phase3o16r_recovery'
TASKS=('binary_qa','multiple_choice_qa','caption')
SHA='5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0';FP='168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b'
GATES={'binary':{'all_required':['margin_increased','normal_accuracy_not_decreased','normal_beats_shuffled_by_one','parseability_not_decreased','output_changed','losses_finite','projector_gradients_nonzero','qwen_gradients_zero','test_access_zero']},'mcq':{'all_required':['margin_increased','image_margin_not_decreased','normal_accuracy_not_decreased','normal_beats_shuffled_by_one','parseability_not_worse','losses_finite','projector_gradients_nonzero','qwen_gradients_zero','test_access_zero']},'caption':{'all_required':['margin_increased','unique_not_decreased','dominant_not_increased','empty_malformed_not_increased','output_changed','prefix_diversity_not_materially_worse','losses_finite','projector_gradients_nonzero','qwen_gradients_zero','test_access_zero']}}
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8')
def maps(rows,split):
 out=[]
 for task in TASKS:
  group=sorted((r for r in rows if r['task_type']==task),key=lambda r:(r['image_id'],r['record_id']))
  for i,r in enumerate(group):
   peer=next(x for x in group[i+1:]+group[:i+1] if x['image_id']!=r['image_id'])
   out.append({'task_type':task,'split':split,'sample_id':r['record_id'],'source_image_id':r['image_id'],'shuffled_sample_id':peer['record_id'],'shuffled_image_id':peer['image_id']})
 return out
def main():
 if OUT.exists():raise RuntimeError('PHASE3O16R_BLOCKED_NAMESPACE_EXISTS')
 manifest=json.loads((RUN/'phase3o12_training_manifest.json').read_text());
 if manifest['test_access_count']!=0:raise RuntimeError('PHASE3O16R_BLOCKED_TEST')
 old_ids=set()
 for task in ('binary','mcq','caption'):
  old=json.loads((OLD/f'phase3o16_{task}_panel.json').read_text());old_ids|={r['record_id'] for x in ('train','validation') for r in old[x]}
 panels={}
 for split,n in (('train',24),('validation',12)):
  panels[split]={}
  for task in TASKS:
   rows=[r for r in sorted(manifest['selection'][split],key=lambda x:x['record_id']) if r['task_type']==task and r['record_id'] not in old_ids]
   panels[split][task]=rows[:n]
   if len(rows[:n])!=n:raise RuntimeError('PHASE3O16R_BLOCKED_COUNT')
 train=[r for v in panels['train'].values() for r in v];val=[r for v in panels['validation'].values() for r in v]
 if {r['image_id'] for r in train}&{r['image_id'] for r in val}:raise RuntimeError('PHASE3O16R_BLOCKED_IMAGE_OVERLAP')
 pre={'phase':'3O.16R','seed':30161,'historical_phase3o16_status':'PHASE3O16_BLOCKED','initial_projector_sha256':SHA,'initial_projector_fingerprint':FP,'qwen_frozen':True,'test_access_count':0,'panel_counts':{'train_per_task':24,'validation_per_task':12},'optimizer':{'name':'AdamW','learning_rate':0.0001,'weight_decay':0.0001,'steps_per_variant':8},'variants':{'binary':['base_ce','smooth_likelihood','low_weight_hinge'],'mcq':['base_ce','target_vs_distractor_ranking','joint_ranking_and_image_margin'],'caption':['base_ce','sequence_likelihood_separation','sequence_likelihood_plus_early_token_image_discrimination']},'success_gates':GATES,'preregistration_locked_before_trial':True}
 pre['preregistration_sha256']=canonical_sha(pre);validate_preregistration(pre)
 OUT.mkdir(parents=True);dump(OUT/'phase3o16r_preregistration.json',pre)
 for task in TASKS:
  n='binary' if task=='binary_qa' else 'mcq' if task=='multiple_choice_qa' else 'caption';dump(OUT/f'phase3o16r_{n}_panel.json',{'task_type':task,'train':panels['train'][task],'validation':panels['validation'][task],'test_access_count':0,'excluded_phase3o16_identities':sorted(old_ids)})
 dump(OUT/'phase3o16r_shuffle_maps.json',{'seed':30161,'test_access_count':0,'maps':{'train':maps(train,'train'),'validation':maps(val,'validation')}})
 print('PHASE3O16R PREREGISTRATION FROZEN');print(json.dumps({'train':{k:len(v) for k,v in panels['train'].items()},'validation':{k:len(v) for k,v in panels['validation'].items()},'excluded_historical_identities':len(old_ids),'test_access_count':0},indent=2))
if __name__=='__main__':main()
