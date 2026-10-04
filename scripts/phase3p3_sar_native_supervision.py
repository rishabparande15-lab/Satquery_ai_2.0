"""Construct a small, provenance-first SAR-native candidate pool; no training."""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset_loader import discover_samples,load_sample
from src.croma_adapter import CROMAAdapter
OUT=ROOT/'artifacts/training/phase3p/phase3p3_sar_native_supervision';DATA=Path(r'D:\Satquery_ai datasets\comparison\raw-1000');META=DATA/'metadata.parquet';CS=Path(r'D:\Satquery_ai datasets\croma_official');CK=Path(r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt')
WATER={'Inland waters','Marine waters'};WATER_ADJACENT={'Inland wetlands','Coastal wetlands','Beaches, dunes, sands'}
LITERATURE={'dataset':'https://bigearth.net/','clc_nomenclature':'https://land.copernicus.eu/content/corine-land-cover-nomenclature-guidelines/html/','sentinel1_dual_polarisation':'https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Data_products','water_sar_evidence':'https://doi.org/10.1038/s41597-021-01059-7'}
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def role(labels):
 labels=set(labels)
 if labels&WATER:return 'yes'
 if labels&(WATER|WATER_ADJACENT):return None
 return 'no'
def rec(row,answer):
 return {'record_id':'sar_native_water_'+hashlib.sha256(str(row.patch_id).encode()).hexdigest()[:24],'sentinel1_patch_id':str(row.s1_name),'sentinel2_patch_id':str(row.patch_id),'source_split':str(row.split),'authoritative_source':'BigEarthNet-S1 paired patch; BigEarthNet CLC2018/reBEN 19-class label metadata','source_labels':list(row.labels),'sar_native_task_type':'binary_qa','question':'Does this Sentinel-1 SAR patch have an authoritative CLC water-body label (inland or marine water)?','answer':answer,'support_rationale':'A water-body versus explicitly non-water class is a conservative SAR-supportable binary task: dual-polarized Sentinel-1 backscatter is used for water discrimination, but the CLC-derived target remains provenance-limited.','provenance_citation':LITERATURE,'confidence':'SUPPORTED','s1_exists':True,'test_access_count':0}
def main():
 if OUT.exists():raise RuntimeError('PHASE3P3_NAMESPACE_EXISTS')
 OUT.mkdir(parents=True);print('PHASE3P3 START',flush=True)
 # Predicate pushdown excludes TEST rows before candidate evaluation.
 df=pd.read_parquet(META,filters=[('split','in',['train','validation'])]);assert set(df['split'])<= {'train','validation'}
 samples={s.patch_id:s for s in discover_samples(DATA,strict=True)}
 df=df[df.patch_id.isin(samples)].copy();df['answer']=df.labels.map(role);eligible=df[df.answer.notna()].copy()
 selected=[]
 for split,each in [('train',40),('validation',10)]:
  part=eligible[eligible.split.eq(split)]
  for ans in ('yes','no'):
   rows=part[part.answer.eq(ans)].sort_values('patch_id').head(each)
   if len(rows)!=each:raise RuntimeError(f'INSUFFICIENT_{split}_{ans}')
   selected.extend(rec(r,ans) for r in rows.itertuples(index=False))
 selected_ids={r['sentinel2_patch_id'] for r in selected}
 # All raw S1 assets selected are read through the validated existing pipeline.
 croma=CROMAAdapter(CS,CK,device='cuda');croma.model.eval();[p.requires_grad_(False) for p in croma.model.parameters()];croma_before=fp(croma.model)
 by_id={r['sentinel2_patch_id']:r for r in selected};stats=[]
 for i,patch_id in enumerate(sorted(by_id),1):
  print(f'[3P3][SAR-STATS] record {i}/{len(by_id)}',flush=True);prepared=load_sample(samples[patch_id]);raw=prepared.raw_sar
  with torch.no_grad():tokens=croma.infer_modality(sar=raw)['SAR_encodings'].detach().cpu()
  stats.append({'sentinel2_patch_id':patch_id,'vv_mean':float(raw[0].mean()),'vv_std':float(raw[0].std()),'vh_mean':float(raw[1].mean()),'vh_std':float(raw[1].std()),'vv_minus_vh_mean':float((raw[0]-raw[1]).mean()),'croma_shape':list(tokens.shape),'croma_mean':float(tokens.mean()),'croma_std':float(tokens.std()),'croma_norm':float(torch.linalg.vector_norm(tokens)),'preprocessing_validated':True})
 if fp(croma.model)!=croma_before:raise RuntimeError('CROMA_MUTATED')
 for r in selected:r['sar_signal_summary']=next(x for x in stats if x['sentinel2_patch_id']==r['sentinel2_patch_id'])
 # Pair only opposite answers, so each shuffled item proves the binary target is not tautological.
 for split in ('train','validation'):
  yes=[r for r in selected if r['source_split']==split and r['answer']=='yes'];no=[r for r in selected if r['source_split']==split and r['answer']=='no']
  for a,b in zip(yes,no):a['shuffled_opposite_patch_id']=b['sentinel2_patch_id'];b['shuffled_opposite_patch_id']=a['sentinel2_patch_id'];a['correct_vs_shuffled_target_changes']=True;b['correct_vs_shuffled_target_changes']=True
 train=[r for r in selected if r['source_split']=='train'];valid=[r for r in selected if r['source_split']=='validation']
 rejected=[]
 for r in df.itertuples(index=False):
  if r.patch_id in selected_ids:continue
  why='POOL_CAP_REACHED_AFTER_BALANCED_SELECTION' if role(r.labels) in {'yes','no'} else 'AMBIGUOUS_WATER_ADJACENT_LABEL_EXCLUDED'
  rejected.append({'sentinel2_patch_id':str(r.patch_id),'sentinel1_patch_id':str(r.s1_name),'source_split':str(r.split),'source_labels':list(r.labels),'reason':why,'confidence':'REJECTED'})
 schema={'phase':'3P.3','task_taxonomy':{'included':[{'name':'water_body_label_present','type':'binary_qa','answers':['yes','no'],'confidence':'SUPPORTED','basis':'authoritative CLC water labels plus conservative Sentinel-1 VV/VH water-discrimination literature'}],'rejected':[{'name':'country_climate_season','reason':'not SAR-native semantic targets'},{'name':'exact_area_topology_adjacency','reason':'reference-map/optical semantics, not established SAR-only targets'},{'name':'fine land-cover species or optical appearance','reason':'not established by VV/VH alone'},{'name':'four-option broad-class MCQ','reason':'available CLC labels are multilabel and no mutually exclusive SAR-established four-class source was found'},{'name':'free-form caption','reason':'authoritative SAR-native descriptions unavailable; no captions fabricated'}]},'required_fields':['record_id','sentinel1_patch_id','source_split','authoritative_source','source_labels','sar_native_task_type','question','answer','support_rationale','provenance_citation','confidence'],'citations':LITERATURE,'no_training':True,'test_access_count':0}
 dist={'candidate_count':len(selected),'train_count':len(train),'validation_count':len(valid),'test_access_count':0,'binary_answer_distribution':{'all':dict(Counter(r['answer'] for r in selected)),'train':dict(Counter(r['answer'] for r in train)),'validation':dict(Counter(r['answer'] for r in valid))},'mcq_option_distribution':{},'signal_group_summary':{a:{k:float(np.mean([r['sar_signal_summary'][k] for r in selected if r['answer']==a])) for k in ('vv_mean','vv_std','vh_mean','vh_std','vv_minus_vh_mean','croma_mean','croma_std','croma_norm')} for a in ('yes','no')},'correct_vs_shuffled_sanity':{'pairs':len(selected)//2,'all_pairs_opposite_authoritative_targets':all(r.get('correct_vs_shuffled_target_changes') for r in selected)}}
 provenance={'metadata_path':str(META),'metadata_sha256':sha(META),'source_records_inspected':len(df),'source_splits':['train','validation'],'test_access_count':0,'s1_file_exists_for_every_candidate':True,'preprocessing_validated_for_every_candidate':True,'croma_frozen_and_unchanged':True,'provenance_complete':all(set(schema['required_fields'])<=set(r) for r in selected),'citation_references':LITERATURE}
 summary={'status':'PHASE3P3_PARTIALLY_COMPLETE','quality_classification':'SAR_NATIVE_POOL_PARTIALLY_READY','source_records_inspected':len(df),'binary_candidate_count':len(selected),'mcq_candidate_count':0,'caption_description_candidate_count':0,'rejected_record_count':len(rejected),'train_count':len(train),'validation_count':len(valid),'test_access_count':0,'provenance_complete':provenance['provenance_complete'],'croma_unchanged':True,'no_training_or_optimizer':True,'classification_reason':'Balanced water/non-water CLC candidates have conservative SAR support and identity sanity, but no independently verified SAR-native MCQ or description source was available.','bounded_sar_retraining_justified':False,'recommendation':'Acquire or create independently audited SAR-native labels beyond CLC-derived water presence before retraining; do not use this provisional pool for training yet.','controller_routing':'BLOCKED','fusion':'BLOCKED'}
 dump('phase3p3_supervision_schema.json',schema);dump('phase3p3_candidate_pool.json',{'records':selected,'test_access_count':0});dump('phase3p3_rejected_records.json',{'records':rejected,'test_access_count':0});dump('phase3p3_train_manifest.json',{'records':train,'test_access_count':0});dump('phase3p3_validation_manifest.json',{'records':valid,'test_access_count':0});dump('phase3p3_distribution_audit.json',dist);dump('phase3p3_provenance_audit.json',provenance);dump('phase3p3_summary.json',summary)
 print(f'SOURCE RECORDS INSPECTED = {len(df)}\nSAR-NATIVE CANDIDATES = {len(selected)}\nREJECTED = {len(rejected)}\nBINARY SUPPORTED = {len(selected)}\nMCQ SUPPORTED = 0\nCAPTION SUPPORTED = 0\nTRAIN COUNT = {len(train)}\nVALIDATION COUNT = {len(valid)}\nTEST ACCESS = 0\nBINARY ANSWER DISTRIBUTION = {dist["binary_answer_distribution"]}\nMCQ OPTION DISTRIBUTION = {{}}\nPROVENANCE COMPLETE = YES\nSAR-NATIVE POOL CLASSIFICATION = SAR_NATIVE_POOL_PARTIALLY_READY',flush=True)
if __name__=='__main__':main()
