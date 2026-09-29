"""Explicit paired BigEarthNet SatQuery v1 demo; no S2 fallback is used."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset_loader import discover_samples,load_sample
from src.satquery_v1 import SatQueryV1Controller
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=os.environ.get('SATQUERY_DATA_ROOT'));p.add_argument('--patch-id',required=True);p.add_argument('--task-type',choices=('binary_qa','multiple_choice_qa','caption'),required=True);p.add_argument('--question',required=True);args=p.parse_args()
 if args.data_root is None: p.error('--data-root or SATQUERY_DATA_ROOT is required')
 samples={x.patch_id:x for x in discover_samples(args.data_root,strict=True)};sample=load_sample(samples[args.patch_id]);result=SatQueryV1Controller().run_satquery(task_type=args.task_type,question=args.question,s1=sample.raw_sar,s2=sample.raw_optical,patch_id=sample.patch_id,s1_patch_id=sample.patch_id,s2_patch_id=sample.patch_id,spatial_metadata=sample.metadata)
 print(json.dumps({'input':{'sentinel_1':'received','sentinel_2':'received','question':args.question,'task':args.task_type},'pipeline':'CROMA_JOINT -> CromaJointProjector -> frozen Qwen','result':result},indent=2))
if __name__=='__main__':main()
