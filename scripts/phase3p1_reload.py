"""Fresh-process Phase 3P.1 validation-loss reproducibility receipt."""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.phase3p1_sar_adaptation import (OUT,DATA,CS,CK,make_tokens,validation_losses,sha)
from src.dataset_loader import discover_samples,load_sample
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT,hash_state
def main():
 m=json.loads((OUT/'phase3p1_training_manifest.json').read_text()); rows=m['selection']['validation']; samples={x.patch_id:x for x in discover_samples(DATA,strict=True)}; prepared={r['image_id']:load_sample(samples[r['image_id']]) for r in rows}
 tokens,croma_fp=make_tokens(prepared); runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda'); model=runtime['model'].eval(); tok=runtime['processor'].tokenizer; freeze_qwen(model); q=hash_state(model)
 p=S1SARProjector().cuda(); p.load_state_dict(torch.load(OUT/'phase3p1_sar_projector_final.pt',map_location='cuda',weights_only=True)['state_dict']); result=validation_losses(model,tok,p,tokens,rows); original=json.loads((OUT/'phase3p1_training_summary.json').read_text())['validation']; delta=abs(result['overall_mean_loss']-original['overall_mean_loss']); agreement='EXACT_MATCH' if delta==0 else ('WITHIN_TOLERANCE' if delta<=1e-6 else 'MISMATCH')
 receipt={'fresh_process':True,'checkpoint_sha256':sha(OUT/'phase3p1_sar_projector_final.pt'),'reload_validation':result,'original_validation_mean':original['overall_mean_loss'],'absolute_difference':delta,'relative_tolerance':0,'absolute_tolerance':1e-6,'agreement':agreement,'qwen_unchanged_during_reload':hash_state(model)==q,'croma_fingerprint':croma_fp,'test_access_count':0}; (OUT/'phase3p1_reload_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n'); print('RELOAD MEAN = '+str(result['overall_mean_loss']),flush=True); print('RELOAD AGREEMENT = '+agreement,flush=True)
 if agreement=='MISMATCH': raise RuntimeError('PHASE3P1_FAILED_REPRODUCIBILITY')
if __name__=='__main__': main()
