"""Read-only standalone Sentinel-1/CROMA-to-frozen-Qwen foundation audit."""
from __future__ import annotations
import gc,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset_loader import discover_samples,load_sample
from src.croma_adapter import CROMAAdapter
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch,freeze_qwen
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT,hash_state
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
OUT=ROOT/'artifacts/training/phase3p/phase3p_sar_foundation';DATA=Path(r'D:\Satquery_ai datasets\comparison\raw-1000');CS=Path(r'D:\Satquery_ai datasets\croma_official');CK=Path(r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt')
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def stats(x):x=np.asarray(x);return {'shape':list(x.shape),'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std()),'nan':int(np.isnan(x).sum()),'inf':int(np.isinf(x).sum())}
def main():
 if OUT.exists():raise RuntimeError('PHASE3P_NAMESPACE_EXISTS')
 samples=discover_samples(DATA,strict=True)[:5];prepared=[load_sample(x) for x in samples];print('[3P][AUDIT] repository SAR path discovered',flush=True);print('SAR INPUT PATH = dataset_loader.load_sample -> normalized VV,VH',flush=True);print('SAR REPRESENTATION PATH = CROMAAdapter.infer_modality(sar) -> SAR_encodings',flush=True);print('SAR TO QWEN PATH = S1SARProjector -> build_qwen_visual_token_batch',flush=True)
 data=[]
 for i,x in enumerate(prepared,1):print(f'[3P][DATA] sample {i}/5',flush=True);data.append({'sample_id':x.patch_id,'normalized':stats(x.sar),'raw':stats(x.raw_sar),'metadata':{k:x.metadata[k] for k in ('crs','resolution','sar_band_order','sar_normalization','sar_resampling')}})
 dump('sar_data_audit.json',{'test_access_count':0,'samples':data})
 c=CROMAAdapter(CS,CK,device='cuda');reps=[]
 for x in prepared:
  y=c.infer_modality(sar=x.raw_sar)['SAR_encodings'];reps.append(y.detach().cpu())
 rs=[]
 for i,r in enumerate(reps):rs.append({'sample_id':prepared[i].patch_id,'shape':list(r.shape),'dtype':str(r.dtype),'mean':float(r.mean()),'std':float(r.std()),'norm':float(torch.linalg.vector_norm(r)),'finite':bool(torch.isfinite(r).all())})
 pairs=[]
 for i in range(len(reps)-1):
  a,b=reps[i].reshape(-1).float(),reps[i+1].reshape(-1).float();pairs.append({'pair':[prepared[i].patch_id,prepared[i+1].patch_id],'l2':float(torch.linalg.vector_norm(a-b)),'cosine':float(torch.nn.functional.cosine_similarity(a[None],b[None]))})
 dump('sar_representation_audit.json',{'test_access_count':0,'representation_path':'official CROMA s1_encoder','records':rs,'pairwise':pairs,'variance':float(torch.stack(reps).float().var())})
 proj=S1SARProjector().cuda().eval();tokens=[proj(r.cuda()).detach().cpu() for r in reps];del c;gc.collect();torch.cuda.empty_cache();
 runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device='cuda');model=runtime['model'].eval();tok=runtime['processor'].tokenizer;freeze=freeze_qwen(model);before=hash_state(model);q=tokens[0].cuda();batch=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=q,questions=['Describe the radar scene.'],answers=['unknown']);out=model(**batch);req={'forward_finite':bool(torch.isfinite(out.logits).all()),'visual_positions':int(batch['input_ids'].eq(model.config.image_token_id).sum()),'attention_valid':bool(batch['attention_mask'].all()),'adapter_shape':list(q.shape),'adapter_finite':bool(torch.isfinite(q).all())};print('QWEN FORWARD = '+('PASS' if req['forward_finite'] else 'FAIL'),flush=True);print('VISUAL TOKEN POSITIONS = '+str(req['visual_positions']),flush=True)
 w=tokens[1].cuda();z=torch.zeros_like(q)
 def f(v):
  b=build_qwen_visual_token_batch(tokenizer=tok,model=model,visual_tokens=v,questions=['Describe the radar scene.'],answers=None);o=model(**b,output_hidden_states=True);return o,b
 co,cb=f(q);so,sb=f(w);zo,zb=f(z);sens={'correct_vs_shuffled_adapter_l2':float(torch.linalg.vector_norm(q-w)),'correct_vs_zero_adapter_l2':float(torch.linalg.vector_norm(q-z)),'correct_vs_shuffled_hidden_l2':float(torch.linalg.vector_norm(co.hidden_states[-1][:,-1]-so.hidden_states[-1][:,-1])),'correct_vs_zero_hidden_l2':float(torch.linalg.vector_norm(co.hidden_states[-1][:,-1]-zo.hidden_states[-1][:,-1])),'correct_vs_shuffled_logit_l2':float(torch.linalg.vector_norm(co.logits[:,-1]-so.logits[:,-1])),'correct_vs_zero_logit_l2':float(torch.linalg.vector_norm(co.logits[:,-1]-zo.logits[:,-1]))};dump('sar_signal_sensitivity.json',{'test_access_count':0,**sens})
 g=generate_with_multimodal_prefix(model,input_ids=cb['input_ids'],inputs_embeds=cb['inputs_embeds'],attention_mask=cb['attention_mask'],image_grid_thw=cb['image_grid_thw'],max_new_tokens=1,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,return_dict_in_generate=True,output_scores=True);delta=float((co.logits[0,-1].float()-g.scores[0][0].float()).abs().max());dump('sar_generation_interface_audit.json',{'test_access_count':0,'max_abs_first_step_delta':delta,'tolerance':1e-3,'pass':delta<=1e-3})
 dump('sar_qwen_forward_audit.json',{'test_access_count':0,'qwen':freeze,'qwen_unchanged':hash_state(model)==before,**req});print('CORRECT-vs-SHUFFLED LOGIT L2 = '+str(sens['correct_vs_shuffled_logit_l2']),flush=True);print('GENERATE/FORWARD FIRST-STEP AGREEMENT = '+str(delta),flush=True)
if __name__=='__main__':main()
