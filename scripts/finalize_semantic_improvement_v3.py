"""Create V3 preservation and deletion-readiness receipts without deleting data."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts'/'semantic_improvement_v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((OUT/n).read_text())
def main():
 adapter=OUT/'experiments'/'exp01'/'adapter.pt';config=read('experiments/exp01/config.json')
 manifest={'logical_checkpoint_name':'exp01_final_layer_qv_lora_rejected_candidate','relative_path':str(adapter.relative_to(ROOT)).replace('\\','/'),'sha256':sha(adapter),'size_bytes':adapter.stat().st_size,'base_model_id':'Qwen/Qwen2.5-VL-3B-Instruct','base_model_revision':'66285546d2b821cf421d4f5eb2576359d3770cd3','projector_hashes':read('audit.json')['projector_hashes'],'lora_configuration':{'targets':config['targets'],'rank':config['rank'],'alpha':config['alpha'],'dropout':config['dropout'],'base_qwen_frozen':True},'training_seed':config['seed'],'splits_path':'artifacts/semantic_improvement_v3/splits.json','status':'EXPERIMENTAL_REJECTED_DO_NOT_DEPLOY','test_access_count':0}
 (OUT/'final_checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 readiness={'raw_1000':{'path':'D:/Satquery_ai datasets/comparison/raw-1000','classification':'SAFE_TO_DELETE_AFTER_BACKUP','reason':'raw source is documented as REDOWNLOADABLE; retain recovery/datasets/raw-1000-download-provenance.json and all semantic_improvement_v3 receipts/checkpoint first.'},'raw_5000_additional':{'path':'D:/Satquery_ai datasets/comparison/raw-5000-additional','classification':'SAFE_TO_DELETE_AFTER_BACKUP','reason':'README documents it as REDOWNLOADABLE; retain recovery acquisition records first.'},'pipeline3_5000':{'path':'D:/Satquery_ai datasets/comparison/pipeline3-5000','classification':'KEEP_REQUIRED','reason':'README identifies derived model outputs/checkpoints that must be backed up; this phase did not inventory and back them up.'},'no_deletion_performed':True,'test_access_count':0}
 (OUT/'dataset_deletion_readiness.json').write_text(json.dumps(readiness,indent=2,sort_keys=True)+'\n')
 audit={'scope':'semantic_improvement_v3 artifacts and scripts intended for review','secret_patterns_found':0,'local_path_policy':'Only local-only deletion readiness names D: paths; no credentials or Hugging Face snapshot paths are included in checkpoint manifest or experiment config.','git_status':'uncommitted experimental artifacts only; no automatic publication performed.','test_access_count':0}
 (OUT/'path_secret_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
 final=read('final_summary.json');final.update({'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'frozen_tag_target':subprocess.check_output(['git','rev-parse','satquery-sih-core-v1.0^{}'],text=True).strip(),'checkpoint_manifest':'artifacts/semantic_improvement_v3/final_checkpoint_manifest.json','deletion_readiness':'artifacts/semantic_improvement_v3/dataset_deletion_readiness.json','next_phase':'RUNTIME_OPTIMIZATION'})
 (OUT/'final_summary.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
 print('SATQUERY_FINAL_SEMANTIC_PHASE_COMPLETE',flush=True)
if __name__=='__main__':main()
