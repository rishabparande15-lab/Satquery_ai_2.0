"""Freeze deterministic task-local Phase 3O.16 panels and maps; no model use."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'artifacts/training/phase3o/phase3o12_run'; OUT=ROOT/'artifacts/training/phase3o/phase3o16_task_specific_objectives'
TASKS=('binary_qa','multiple_choice_qa','caption')
PREREG={'phase':'3O.16','seed':3016,'initial_projector_sha256':'5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0','initial_projector_fingerprint':'168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b','qwen_frozen':True,'test_access_count':0,'panels':{'train_per_task':24,'validation_per_task':12},'budget':'eight equal optimizer steps per preregistered task-local variant; no full training','binary_variants':['base_ce','smooth_sequence_likelihood','low_weight_hinge'],'mcq_variants':['base_ce','target_vs_distractor_ranking','joint_ranking_and_image_margin'],'caption_variants':['base_ce','sequence_likelihood_separation','sequence_likelihood_plus_early_token_image_discrimination'],'success_gates':'task-local validation criteria specified in Phase 3O.16 handoff; no cross-task aggregate selection'}
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def maps(rows,split):
 out=[]
 for task in TASKS:
  g=sorted((r for r in rows if r['task_type']==task),key=lambda r:(r['image_id'],r['record_id']))
  for i,r in enumerate(g):
   q=next(x for x in g[i+1:]+g[:i+1] if x['image_id']!=r['image_id']);out.append({'task_type':task,'split':split,'sample_id':r['record_id'],'source_image_id':r['image_id'],'shuffled_sample_id':q['record_id'],'shuffled_image_id':q['image_id']})
 return out
def main():
 if OUT.exists():raise RuntimeError('PHASE3O16_BLOCKED_NAMESPACE_EXISTS')
 m=json.loads((RUN/'phase3o12_training_manifest.json').read_text())
 if m['test_access_count'] or any(r['split']=='test' for s in m['selection'].values() for r in s):raise RuntimeError('PHASE3O16_BLOCKED_TEST')
 panels={}
 for split,n in [('train',24),('validation',12)]:
  panels[split]={t:sorted((r for r in m['selection'][split] if r['task_type']==t),key=lambda r:r['record_id'])[:n] for t in TASKS}
  if any(len(x)!=n for x in panels[split].values()):raise RuntimeError('PHASE3O16_BLOCKED_COUNT')
 train=[r for x in panels['train'].values() for r in x];val=[r for x in panels['validation'].values() for r in x]
 if {r['image_id'] for r in train}&{r['image_id'] for r in val}:raise RuntimeError('PHASE3O16_BLOCKED_OVERLAP')
 sm={'train':maps(train,'train'),'validation':maps(val,'validation')}
 if any(x['source_image_id']==x['shuffled_image_id'] for v in sm.values() for x in v):raise RuntimeError('PHASE3O16_BLOCKED_SELF_MAP')
 OUT.mkdir(parents=True);dump(OUT/'phase3o16_preregistration.json',PREREG)
 for task in TASKS:
  name='binary' if task=='binary_qa' else ('mcq' if task=='multiple_choice_qa' else task)
  dump(OUT/f'phase3o16_{name}_panel.json',{'task_type':task,'train':panels['train'][task],'validation':panels['validation'][task],'test_access_count':0})
 dump(OUT/'phase3o16_shuffle_maps.json',{'seed':3016,'test_access_count':0,'maps':sm})
 print(json.dumps({'status':'PHASE3O16_PANELS_PREREGISTERED','train':{k:len(v) for k,v in panels['train'].items()},'validation':{k:len(v) for k,v in panels['validation'].items()},'test_access_count':0},indent=2))
if __name__=='__main__':main()
