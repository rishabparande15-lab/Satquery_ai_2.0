"""Freeze the Phase 3O.18 prior/prefix intervention protocol before training."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/training/phase3o/phase3o12_run';OUT=ROOT/'artifacts/training/phase3o/phase3o18_prior_prefix_intervention';sys.path.insert(0,str(ROOT))
from src.eo_vlm.phase3o16r_protocol import canonical_sha
SHA='5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0';FP='168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b'
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def mapping(rows,task):
 out=[]
 for i,r in enumerate(rows):
  q=next(x for x in rows[i+1:]+rows[:i+1] if x['image_id']!=r['image_id']);out.append({'task':task,'sample_id':r['record_id'],'shuffled_sample_id':q['record_id'],'source_image_id':r['image_id'],'shuffled_image_id':q['image_id']})
 return out
def main():
 if OUT.exists():raise RuntimeError('PHASE3O18_NAMESPACE_EXISTS')
 m=json.loads((RUN/'phase3o12_training_manifest.json').read_text());assert m['test_access_count']==0
 # Exclude Phase 3O.17 evaluation identities first. Earlier 3O.16/16R identities
 # must be reused because 50 validation examples cannot supply 16 otherwise.
 p17=json.loads((ROOT/'artifacts/training/phase3o/phase3o17_decision_prior_diagnostic/phase3o17_panels.json').read_text());used={r['record_id'] for xs in p17['panels'].values() for r in xs}
 panels={}
 for task,name in [('multiple_choice_qa','mcq'),('caption','caption')]:
  tr=[r for r in sorted(m['selection']['train'],key=lambda x:x['record_id']) if r['task_type']==task][:32]
  va=[r for r in sorted(m['selection']['validation'],key=lambda x:x['record_id']) if r['task_type']==task and r['record_id'] not in used][:16]
  if len(tr)!=32 or len(va)!=16 or {r['image_id'] for r in tr}&{r['image_id'] for r in va}:raise RuntimeError('PHASE3O18_PANEL_FAILURE')
  panels[name]={'train':tr,'validation':va,'prior_identity_reuse_due_to_pool_limit':True}
 pre={'phase':'3O.18','seed':30180,'initial_projector_sha256':SHA,'initial_projector_fingerprint':FP,'test_access_count':0,'qwen_frozen':True,'panels':{'mcq':{'train':32,'validation':16},'caption':{'train':32,'validation':16}},'optimizer':{'name':'AdamW','learning_rate':0.0001,'weight_decay':0.0001,'steps':8},'objectives':{'mcq_base_ce':'CE','mcq_target_vs_distractor':'CE + 0.25*max(0,0.10-(S_target_correct-max(S_distractor_correct)))','mcq_joint':'CE + 0.25*max(0,0.10-(S_target_correct-max(S_distractor_correct))) + 0.25*softplus(-(S_target_correct-S_target_shuffled))','caption_base_ce':'CE','caption_early_token':'CE + 0.25*mean_{positions 1..5} softplus(-(logP(ref_t|correct)-logP(ref_t|shuffled)))','caption_prefix_weighted':'CE + 0.35*early_prefix_separation(1..5) + 0.10*later_sequence_separation'},'gates':{'mcq':['validation_margin_improves','target_rank1_improves','dominant_option_not_worse','normal_accuracy_not_decreases','ranking_distinguishability_improves','parseability_not_worse','finite','projector_gradient_nonzero','qwen_gradient_zero','test_access_zero'],'caption':['early_discrimination_improves','early_diversity_not_decreases','complete_diversity_not_decreases','dominant_prefix_not_worse','shuffle_changes_present','empty_not_increase','finite','projector_gradient_nonzero','qwen_gradient_zero','test_access_zero']},'checkpoint_persistence':'every variant before post evaluation','validation_pool_constraint':'Phase3O17 identities excluded; Phase3O16/16R reuse is necessary to obtain 16 validation records from each 50-record task pool.'}
 pre['preregistration_sha256']=canonical_sha(pre);OUT.mkdir(parents=True);dump(OUT/'phase3o18_preregistration.json',pre);dump(OUT/'phase3o18_mcq_panels.json',panels['mcq']);dump(OUT/'phase3o18_caption_panels.json',panels['caption']);dump(OUT/'phase3o18_shuffle_maps.json',{'mcq':{k:mapping(v,'mcq') for k,v in panels['mcq'].items() if k in ('train','validation')},'caption':{k:mapping(v,'caption') for k,v in panels['caption'].items() if k in ('train','validation')},'test_access_count':0});print('PHASE3O18 PREREGISTRATION FROZEN');print(json.dumps({'mcq':{k:len(v) for k,v in panels['mcq'].items() if isinstance(v,list)},'caption':{k:len(v) for k,v in panels['caption'].items() if isinstance(v,list)},'test_access_count':0},indent=2))
if __name__=='__main__':main()
