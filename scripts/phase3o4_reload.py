"""Standalone fresh-process Phase 3O.3 checkpoint reload validation; never trains."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,psutil,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_phase3o2_adaptation_pilot import MODEL_SNAPSHOT,MODEL_REVISION,ROOTS,cube,file_index,hash_state
from src.eo_vlm.s2_asset_gate import S2_BANDS,inspect_area,sha256_file
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.training import build_qwen_token_batch,freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
OUT=ROOT/'artifacts/training/phase3o/phase3o5_run'; CK=OUT/'s2_multispectral_projector.pt'; EXPECTED='e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db'; FINGERPRINT='db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46'
def state_fp(state):
 h=hashlib.sha256()
 for n,v in sorted(state.items()):h.update(n.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 ckhash=sha256_file(CK); state=torch.load(CK,map_location='cpu',weights_only=False)['state_dict'];fp=state_fp(state)
 if ckhash!=EXPECTED or fp!=FINGERPRINT: raise RuntimeError('CHECKPOINT_INTEGRITY_FAILED')
 pilot=json.loads((OUT/'pilot_manifest.json').read_text()); validation=pilot['selection']['validation']; train={x['image_id'] for x in pilot['selection']['train']}
 if len(validation)!=20 or train & {x['image_id'] for x in validation}:raise RuntimeError('VALIDATION_MANIFEST_INTEGRITY_FAILED')
 idx=file_index(); records={}
 for line in (ROOT/'artifacts/annotations/image_language/visual_vqa_candidates.jsonl').open():
  r=json.loads(line);records[r['record_id']]=r
 assets=[]
 for r in validation:
  row=records[r['record_id']]; origin=Path(r['source_root']); paths={b:next(origin.rglob(f"{row['image_id']}_{b}.tif")) for b in S2_BANDS}; asset=inspect_area(row,paths).to_dict()
  historical_hash=hashlib.sha256(json.dumps([(b,sha256_file(paths[b])) for b in S2_BANDS],separators=(',',':')).encode()).hexdigest()
  if historical_hash!=r['s2_source_hash']:raise RuntimeError('VALIDATION_MANIFEST_INTEGRITY_FAILED')
  assets.append((row,asset))
 process=psutil.Process();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();adapter=Qwen25VLRGBAdapter();runtime=adapter.load_model(MODEL_SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'];model.eval();freeze=freeze_qwen(model)
 projector=S2MultispectralProjector().to('cuda');projector.load_state_dict(state);before_p=hash_state(projector);before_q=hash_state(model);losses=[];valid_start=time.perf_counter()
 with torch.no_grad():
  for row,asset in assets:
   b=build_qwen_token_batch(tokenizer=runtime['processor'].tokenizer,model=model,projector=projector,s2=cube(asset),questions=[row['question']],answers=[row['answer']]);loss=model(**b).loss;losses.append({'sample_id':row['record_id'],'task_type':row['task_type'],'loss':float(loss.cpu()),'finite_loss':bool(torch.isfinite(loss))})
 mean=float(np.mean([x['loss'] for x in losses]));diff=abs(mean-0.5193666338920593);after_p=hash_state(projector);after_q=hash_state(model);receipt={'phase':'3O.5','status':'PHASE3O5_RELOAD_VALIDATED' if diff<=1e-6 and before_p==after_p and before_q==after_q else 'PHASE3O5_RELOAD_FAILED','fresh_process':True,'checkpoint_sha256':ckhash,'expected_checkpoint_sha256':EXPECTED,'checkpoint_sha256_match':True,'model_state_fingerprint':fp,'expected_model_state_fingerprint':FINGERPRINT,'model_state_match':True,'base_model':adapter.model_id,'base_model_revision':MODEL_REVISION,'qwen_frozen':freeze['qwen_trainable_parameter_count']==0,'projector_parameter_count':sum(p.numel() for p in projector.parameters()),'validation_count':20,'test_samples_accessed':0,'per_sample_validation':losses,'historical_validation_mean':0.5193666338920593,'reload_validation_mean':mean,'comparison':{'absolute_tolerance':1e-6,'relative_tolerance':0,'absolute_difference':diff,'agreement_status':'EXACT_MATCH' if diff==0 else ('NUMERICALLY_MATCHING' if diff<=1e-6 else 'MISMATCH')},'validation_side_effect_free':before_p==after_p and before_q==after_q,'reload_duration_seconds':time.perf_counter()-started,'validation_duration_seconds':time.perf_counter()-valid_start,'peak_vram_bytes':int(torch.cuda.max_memory_allocated()),'peak_ram_bytes':process.memory_info().rss,'scientific_baseline_changed':False,'benchmark_inference_executed':False,'blocking_reasons':[]};(OUT/'phase3o5_reload_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
