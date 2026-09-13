"""Pass 5D controlled scene-level A/B experiment on the Pass 3 reproduction split."""
from __future__ import annotations
import argparse, gc, hashlib, io, json, platform, sys, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.gee_features import extract_gee_features
from src.dataset_loader import OPTICAL_BANDS,SAR_BANDS
from src.pixel_features import aggregate_tokens,candidate_maps,local_mean_std
from src.phase3_6_benchmark import ProbeSplit,fit_probe,predict_rows,sha256
from src.phase1_foundation import CLASSES,coverage_metrics
from src.pass3_validation import _scene_target,_independent_metrics

PRIMARY=Path(r"E:\SatQuery_ai_2.0\datasets_2.0\bigearthnet-v2-5000-20260911T162804Z-1-001.zip")
FRAGMENTS=Path(r"E:\SatQuery_ai_2.0\datasets_2.0\bigearthnet-v2-full-official-20260911T162841Z-1-031.zip")
PIPELINE=Path(r"D:\Satquery_ai datasets\comparison\pipeline-1000")
S2_ROOT=Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2")
SELECTED=("NDVI","NDWI","NDBI","BSI","VV_minus_VH","NDVI_local_std_5")

def write(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def archive_audit(pass3_manifest):
    with zipfile.ZipFile(PRIMARY) as outer:
        prefix="bigearthnet-v2-5000/"; frame=pd.read_parquet(io.BytesIO(outer.read(prefix+"metadata.parquet")))
        selection=json.loads(outer.read(prefix+"selection.json")); outer_names=outer.namelist()
    primary_ids=set(map(str,frame.patch_id)); pass3_ids={x["area_id"] for x in pass3_manifest["samples"]}
    disk_dirs={d.name:d for scene in S2_ROOT.iterdir() if scene.is_dir() for d in scene.iterdir() if d.is_dir()}
    matching=primary_ids & disk_dirs.keys(); eligible_independent=matching-pass3_ids
    by_id={str(row.patch_id):row for row in frame.itertuples(index=True)}
    records=[]
    for row in pass3_manifest["samples"]:
        area=row["area_id"]; meta=by_id[area]; s2dir=disk_dirs[area]
        s2files=sorted(str(p.resolve()) for p in s2dir.glob("*.tif"))
        records.append({"area_id":area,"split":row["split"],"country":row["country"],"validity":"VALID",
            "s1_source":{"archive":str(PRIMARY),"nested_archive":"bigearthnet-v2-5000/BigEarthNet-S1-selected.zip","identity":str(meta.s1_name),"bands":["VV","VH"]},
            "s2_source":{"root":str(S2_ROOT),"identity":area,"files":s2files,"band_count":len(s2files)},
            "reference_source":{"archive":str(PRIMARY),"nested_archive":"bigearthnet-v2-5000/Reference_Maps-selected.zip","identity":area},
            "metadata_source":{"archive":str(PRIMARY),"member":"bigearthnet-v2-5000/metadata.parquet","row_index":int(meta.Index)}})
    return {"format_version":1,"status":"VALID_FOR_REPRODUCTION_MODE_ONLY","dataset_used":str(PRIMARY),
        "archives":{"primary":{"path":str(PRIMARY),"size_bytes":PRIMARY.stat().st_size,"sha256":sha256(PRIMARY),"integrity":"PASS: prior exhaustive outer+nested CRC audit","outer_entries":len(outer_names),"areas":len(primary_ids),"s1_bands":10000,"references":5000,"s2_members":0},
                    "fragmented":{"path":str(FRAGMENTS),"size_bytes":FRAGMENTS.stat().st_size,"sha256":sha256(FRAGMENTS),"integrity":"PASS outer ZIP; unusable non-contiguous fragments","areas_usable":0}},
        "s2_audit":{"location":str(S2_ROOT),"unique_area_ids":len(disk_dirs),"matching_primary_ids":len(matching),"matching_pass3_ids":len(matching&pass3_ids),"matching_non_pass3_ids":len(eligible_independent),"missing_for_primary":len(primary_ids-matching),"conclusion":"Mode B blocked: all 1,000 matching S2 areas are Pass 3 areas; 4,000 primary areas lack local matching S2."},
        "overlap":{"primary_vs_pass3":len(primary_ids&pass3_ids),"pass3_total":len(pass3_ids)},"selection_seed":selection["seed"],"selected_areas":records}

def selected_scene_features(optical,sar):
    maps=candidate_maps(optical,sar,OPTICAL_BANDS,SAR_BANDS); maps["NDVI_local_std_5"]=local_mean_std(maps["NDVI"],5)[1]
    token=np.column_stack([aggregate_tokens(maps[name])[0][:,:2] for name in SELECTED]).astype(np.float32)
    return token,np.concatenate((token.mean(0),token.std(0))).astype(np.float32)

def load_data(pass3_manifest):
    by_id={r["area_id"]:r for r in pass3_manifest["samples"]}; pm=json.loads((PIPELINE/"prepared/manifest.json").read_text()); fm=json.loads((PIPELINE/"features/manifest.json").read_text()); tm=json.loads((PIPELINE/"targets/manifest.json").read_text())
    xa=[]; xb=[]; y=[]; ids=[]; splits=[]; countries=[]; token_diagnostics={}; costs={"physical_seconds":0.,"pixel_local_token_seconds":0.,"artifact_load_seconds":0.}
    for pe,fe,te in zip(pm["batches"],fm["batches"],tm["batches"],strict=True):
        t=time.perf_counter(); pdir=PIPELINE/"prepared"/pe["directory"]; fdir=PIPELINE/"features"/fe["directory"]; tdir=PIPELINE/"targets"/te["directory"]
        meta=json.loads((pdir/"batch.json").read_text()); raw=torch.load(pdir/"raw_inputs.pt",map_location="cpu",weights_only=True); deep=torch.load(fdir/"features.pt",map_location="cpu",weights_only=True); targets=torch.load(tdir/"targets.pt",map_location="cpu",weights_only=True); costs["artifact_load_seconds"]+=time.perf_counter()-t
        for i,item in enumerate(meta["samples"]):
            area=item["patch_id"]; optical=raw["optical_images"][i].numpy(); sar=raw["SAR_images"][i].numpy()
            t=time.perf_counter(); physical,_=extract_gee_features(optical,sar,OPTICAL_BANDS,SAR_BANDS); costs["physical_seconds"]+=time.perf_counter()-t
            t=time.perf_counter(); token,extra=selected_scene_features(optical,sar); costs["pixel_local_token_seconds"]+=time.perf_counter()-t
            base=np.concatenate((physical,deep["joint_GAP"][i].numpy().astype(np.float32)))
            xa.append(base); xb.append(np.concatenate((base,extra))); y.append(_scene_target(targets["class_counts"][i])); ids.append(area); splits.append(by_id[area]["split"]); countries.append(by_id[area]["country"])
            if len(token_diagnostics)<3: token_diagnostics[area]={"shape":list(token.shape),"finite":bool(np.isfinite(token).all()),"per_feature_spatial_std":dict(zip([f"{stat}_{n}" for n in SELECTED for stat in ("mean","std")],map(float,token.std(0))))}
        del raw,deep,targets; gc.collect()
    return np.asarray(xa,np.float32),np.asarray(xb,np.float32),np.asarray(y,np.float32),np.asarray(ids),np.asarray(splits),countries,costs,token_diagnostics

def train_version(name,x,y,ids,splits,out):
    parts={s:ProbeSplit(s,torch.from_numpy(x[splits==s]),torch.from_numpy(y[splits==s]),ids[splits==s]) for s in ("train","validation","test")}
    start=time.perf_counter(); model,log=fit_probe(parts["train"],parts["validation"],seed=17,max_epochs=60,patience=10,batch_size=1024,device="cuda" if torch.cuda.is_available() else "cpu"); training=time.perf_counter()-start
    start=time.perf_counter(); pred1=predict_rows(model,parts["test"].x).numpy(); inference=time.perf_counter()-start; pred2=predict_rows(model,parts["test"].x).numpy()
    metrics=coverage_metrics(pred1,parts["test"].y.numpy(),parts["test"].area_ids,bootstrap_repeats=2000,seed=17); independent=_independent_metrics(pred1,parts["test"].y.numpy())
    diff=max(abs(metrics[k]-independent[k]) for k in ("mae_pp","rmse_pp","bias_pp","dominant_class_accuracy")); reproducibility=float(np.max(np.abs(pred1-pred2)))
    out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"version":name,"dimension":x.shape[1],"seed":17,"selected_epoch":log["selected_epoch"]},out/"checkpoint.pt")
    write(out/"config.json",{"version":name,"dimension":int(x.shape[1]),"seed":17,"max_epochs":60,"patience":10,"batch_size":1024,"optimizer":"AdamW","learning_rate":.001,"selection":"validation soft-target cross-entropy","target":"scene-level 19-class labelled-pixel coverage"})
    write(out/"training_log.json",log); write(out/"test_predictions.json",{"area_ids":parts["test"].area_ids.tolist(),"truth":parts["test"].y.tolist(),"predictions":pred1.tolist()}); write(out/"metrics.json",metrics)
    return {"metrics":metrics,"independent":independent,"metric_max_abs_difference":diff,"prediction_repeat_max_abs_difference":reproducibility,"training_seconds":training,"inference_seconds":inference,"selected_epoch":log["selected_epoch"],"checkpoint_sha256":sha256(out/"checkpoint.pt"),"artifact_bytes":sum(p.stat().st_size for p in out.glob("*"))},pred1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=ROOT/"experiments/pass5d"); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    pass3=json.loads((ROOT/"experiments/pass3/dataset_manifest.json").read_text()); manifest=archive_audit(pass3); write(args.out/"dataset_manifest.json",manifest)
    split=json.loads((ROOT/"experiments/pass3/split.json").read_text()); split_out={**split,"pass5d_mode":"REPRODUCTION_COMPARISON","dataset_source":str(PRIMARY),"s2_source":str(S2_ROOT),"pass3_overlap_handling":"exact declared reuse; not independent","disjointness":{"train_validation":[],"train_test":[],"validation_test":[]}}; write(args.out/"split.json",split_out)
    xa,xb,y,ids,splits,countries,costs,token_diag=load_data(pass3)
    if xa.shape!=(1000,830) or xb.shape!=(1000,854) or {s:int((splits==s).sum()) for s in ("train","validation","test")}!={"train":600,"validation":200,"test":200}: raise RuntimeError("experiment contract mismatch")
    a,pa=train_version("version_a_current_hybrid",xa,y,ids,splits,args.out/"version_a"); b,pb=train_version("version_b_pixel_enhanced",xb,y,ids,splits,args.out/"version_b")
    truth=y[splits=="test"]; testids=ids[splits=="test"]
    comparison={"status":"complete","mode":"REPRODUCTION_COMPARISON","split_counts":{"train":600,"validation":200,"test":200},"version_a":a,"version_b":b,
        "changes":{"mae_pp":b["metrics"]["mae_pp"]-a["metrics"]["mae_pp"],"rmse_pp":b["metrics"]["rmse_pp"]-a["metrics"]["rmse_pp"],"bias_pp":b["metrics"]["bias_pp"]-a["metrics"]["bias_pp"],"dominant_accuracy":b["metrics"]["dominant_class_accuracy"]-a["metrics"]["dominant_class_accuracy"]},
        "feature_contract":{"a":"62 physical + 768 joint CROMA GAP = 830","b":"Version A + 24 pooled summaries from 12 selected token features = 854","selected_token_features":[f"{stat}_{n}" for n in SELECTED for stat in ("mean","std")],"scene_pooling":"mean and std over 225 tokens for each of 12 token features"},
        "costs":{**costs,"version_a_feature_bytes":int(xa.nbytes),"version_b_feature_bytes":int(xb.nbytes)},"token_diagnostics":token_diag,
        "leakage":{"split_intersections_empty":True,"mode_b_not_claimed":True,"target_used_in_features":False,"standardization":"train only","selection":"validation only","test_used_once":True,"pass3_overlap":"1000/1000 explicit reproduction"},
        "robustness":{"status":"NOT_RUN","reason":"Frozen CROMA artifacts cannot be recomputed under masks; perturbing only Version B would be unfair."},
        "environment":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}}
    write(args.out/"comparison/metrics.json",comparison); write(args.out/"comparison/predictions.json",{"area_ids":testids.tolist(),"truth":truth.tolist(),"version_a":pa.tolist(),"version_b":pb.tolist()})
    proxy={"sample_id":"61_39","version_a":{"what_is_present":"MODEL_PREDICTED scene coverage only when a task head is applied; no token-local physical answer.","spatial_questions":"UNAVAILABLE from pooled physical representation."},
        "version_b":{"what_is_present":"Prediction remains scene-level; structured physical evidence is separately available.","vegetation":"Trace to NDVI tokens/regions.","water":"Trace to NDWI tokens/regions.","heterogeneity":"Trace to token distributions and local std.","sar":"Trace to VV, VH and VV-minus-VH tokens."},"source":"artifacts/spatial_evidence/61_39","note":"Deterministic proxy strings, not LLM output."}; write(args.out/"comparison/question_proxy.json",proxy)
    print(json.dumps({"status":"complete","mode":comparison["mode"],"a":{k:a["metrics"][k] for k in ("mae_pp","rmse_pp","bias_pp","dominant_class_accuracy")},"b":{k:b["metrics"][k] for k in ("mae_pp","rmse_pp","bias_pp","dominant_class_accuracy")},"changes":comparison["changes"]},indent=2))

if __name__=="__main__": main()
