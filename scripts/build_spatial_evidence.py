"""Build the deterministic Pass 5C evidence artifact for real sample 61_39."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.evidence_schema import build_spatial_evidence, canonical_sha256, file_sha256
from src.pixel_features import safe_normalized_difference


def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,default=ROOT/"experiments/outputs/final_audit/sample-61_39"); ap.add_argument("--out",type=Path,default=ROOT/"artifacts/spatial_evidence/61_39"); args=ap.parse_args()
    receipt=json.loads((args.source/"receipt.json").read_text()); arrays=receipt["arrays"]; provenance0=receipt["provenance"]
    optical=np.load(args.source/arrays["raw_optical"]["file"],allow_pickle=False); sar=np.load(args.source/arrays["raw_sar"]["file"],allow_pickle=False)
    acquisition=provenance0["acquisition"]; croma={}
    for mode,key in (("optical","croma_optical_encodings"),("sar","croma_SAR_encodings"),("joint","croma_joint_encodings")):
        meta=arrays[key]; croma[mode]={"path":str((args.source/meta["file"]).resolve()),"sha256":meta["sha256"],"shape":meta["shape"],"dtype":meta["dtype"],"indexing":"array[token_index, :]"}
    provenance={"dataset":"BigEarthNet v2 verified three-sample extract","sample_id":"61_39","source_artifact":str(args.source.resolve()),
        "source_receipt_sha256":file_sha256(args.source/"receipt.json"),"pixel_source_artifact":str((ROOT/"artifacts/pixel_feature_probe/61_39/receipt.json").resolve()),
        "source_hashes":acquisition["source_hashes"],"preprocessing_version":provenance0["phase1"]["normalization_profile"],
        "model_version":{"croma_checkpoint_sha256":provenance0["phase2"]["provenance"]["croma"]["checkpoint_sha256"],
            "task_prediction":"UNAVAILABLE; no trained task head for 61_39"},"timestamp":provenance0["execution_plan"]["created_at"]}
    kwargs=dict(sample_id="61_39",optical=optical,sar=sar,transform=acquisition["transform"],crs=acquisition["crs"],
        scene_metadata=provenance0["scene"],provenance=provenance,croma=croma,predictions=None)
    first=build_spatial_evidence(**kwargs); second=build_spatial_evidence(**kwargs)
    h1,h2=canonical_sha256(first),canonical_sha256(second)
    if h1 != h2: raise RuntimeError("evidence build is not deterministic")
    args.out.mkdir(parents=True,exist_ok=True)
    scene={"schema_version":first["schema_version"],"scene":first["scene"],"sensor_views":first["sensor_views"],"claims":first["claims"]}
    write_json(args.out/"scene.json",scene); write_json(args.out/"tokens.json",first["tokens"]); write_json(args.out/"regions.json",first["regions"])
    write_json(args.out/"provenance.json",{**first["provenance"],"evidence_sha256":h1,"determinism":{"runs":2,"canonical_sha256_run_1":h1,"canonical_sha256_run_2":h2,"tolerance":"exact JSON equality; float calculations deterministic on this NumPy/runtime"}})
    write_json(args.out/"evidence.json",first)
    ndvi=safe_normalized_difference(optical[7],optical[3])[0]; rgb=np.stack((optical[3],optical[2],optical[1]),-1); lo,hi=np.percentile(rgb,(2,98)); rgb=np.clip((rgb-lo)/(hi-lo),0,1)
    vegetation={i for r in first["regions"] if r["feature_id"]=="NDVI" for i in r["token_indices"]}; mask=np.zeros((15,15));
    for i in vegetation: mask.flat[i]=1
    rgb8=(rgb*255).astype(np.uint8); grid_rgb=rgb8.copy()
    for v in range(0,120,8): grid_rgb[v:v+1,:,:]=255; grid_rgb[:,v:v+1,:]=255
    scaled=np.clip((ndvi+1)/2,0,1); ndvi_rgb=np.stack((255*(1-scaled),255*scaled,80*np.ones_like(scaled)),-1).astype(np.uint8)
    region_mask=mask.repeat(8,0).repeat(8,1)[...,None]; region_rgb=np.where(region_mask>0,(.55*rgb8+.45*np.array([20,230,40])).astype(np.uint8),rgb8)
    composite=np.concatenate((grid_rgb,ndvi_rgb,region_rgb),axis=1)
    with rasterio.open(args.out/"diagnostic.png","w",driver="PNG",width=composite.shape[1],height=composite.shape[0],count=3,dtype="uint8") as dst:
        dst.write(composite.transpose(2,0,1))
    print(json.dumps({"status":"complete","tokens":len(first["tokens"]),"regions":len(first["regions"]),"claims":len(first["claims"]),"sha256":h1,"determinism":"PASS"},indent=2))

if __name__=="__main__": main()
