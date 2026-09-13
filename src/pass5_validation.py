"""Leakage-controlled SAR-only generalization experiment for Pass 5.

The input archive is treated as read-only.  All split decisions are made and
audited before raster decoding, representation extraction, or training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import random
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import rasterio
import torch

from .config import get_settings
from .croma_adapter import CROMAAdapter
from .modality_features import extract as extract_physical
from .phase1_foundation import CLASSES, SAR_BANDS, coverage_metrics, patch_label_targets
from .phase3_6_benchmark import ProbeSplit, fit_probe, predict_rows
from .scientific_artifacts import save_artifact_bundle, verify_artifact_bundle

ARCHIVE_MEMBER_ROOT = "bigearthnet-v2-5000"
EXPECTED_ARCHIVE_NAME = "bigearthnet-v2-5000-20260911T162804Z-1-001.zip"
EXPECTED_ARCHIVE_SHA256 = "b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595"


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_index(names: list[str], suffix: str) -> dict[str, str]:
    found = {}
    for name in names:
        if name.endswith(suffix):
            identity = PurePosixPath(name).parent.name
            if identity in found:
                raise ValueError(f"duplicate identity for {suffix}: {identity}")
            found[identity] = name
    return found


def build_gates(archive: Path, pass3_manifest: Path, output: Path, seed: int = 53) -> tuple[dict, dict]:
    """Build manifest/exclusion/split and prove leakage is absent before training."""
    if archive.name != EXPECTED_ARCHIVE_NAME:
        raise ValueError(f"Pass 5 requires exactly {EXPECTED_ARCHIVE_NAME}")
    archive_hash = _sha256(archive)
    if archive_hash.lower() != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"dataset archive SHA-256 mismatch: {archive_hash}")
    pass3 = json.loads(pass3_manifest.read_text(encoding="utf-8"))
    pass3_rows = pass3["samples"]
    exclusion = {
        "format_version": 1,
        "source_manifest": str(pass3_manifest),
        "source_manifest_sha256": _sha256(pass3_manifest),
        "count": len(pass3_rows),
        "area_ids": sorted(row["area_id"] for row in pass3_rows),
        "s1_identities": sorted(row["s1_identity"] for row in pass3_rows),
        "metadata_identities": sorted({row["s2_identity"] for row in pass3_rows}),
    }
    if any(len(exclusion[key]) != 1000 for key in ("area_ids", "s1_identities", "metadata_identities")):
        raise ValueError("Pass-3 exclusion identities must each contain exactly 1,000 unique values")
    _json(output / "pass3_exclusion_set.json", exclusion)

    with zipfile.ZipFile(archive) as outer:
        metadata = pd.read_parquet(io.BytesIO(outer.read(f"{ARCHIVE_MEMBER_ROOT}/metadata.parquet")))
        with tempfile.TemporaryDirectory(prefix="satquery-pass5-") as temp:
            s1_path, ref_path = Path(temp) / "s1.zip", Path(temp) / "reference.zip"
            s1_path.write_bytes(outer.read(f"{ARCHIVE_MEMBER_ROOT}/BigEarthNet-S1-selected.zip"))
            ref_path.write_bytes(outer.read(f"{ARCHIVE_MEMBER_ROOT}/Reference_Maps-selected.zip"))
            with zipfile.ZipFile(s1_path) as s1z, zipfile.ZipFile(ref_path) as refz:
                s1_names, ref_names = s1z.namelist(), refz.namelist()
                vv = _member_index(s1_names, "_VV.tif")
                vh = _member_index(s1_names, "_VH.tif")
                refs = _member_index(ref_names, "_reference_map.tif")
    required = {"patch_id", "s1_name", "country", "split", "s2v1_name"}
    if not required.issubset(metadata.columns) or len(metadata) != 5000:
        raise ValueError("unexpected metadata contract")
    if metadata.patch_id.duplicated().any() or metadata.s1_name.duplicated().any():
        raise ValueError("duplicate canonical metadata identity")
    p3_area, p3_s1, p3_meta = map(set, (exclusion["area_ids"], exclusion["s1_identities"], exclusion["metadata_identities"]))
    samples = []
    for row in metadata.itertuples(index=False):
        area, s1 = str(row.patch_id), str(row.s1_name)
        reasons = []
        if s1 not in vv: reasons.append("missing VV")
        if s1 not in vh: reasons.append("missing VH")
        if area not in refs: reasons.append("missing reference")
        matches = sorted(set((["area_id"] if area in p3_area else []) + (["s1_identity"] if s1 in p3_s1 else []) + (["metadata_identity"] if area in p3_meta else [])))
        samples.append({"area_id": area, "s1_identity": s1, "metadata_identity": area,
                        "s2v1_identity": str(row.s2v1_name), "country": str(row.country),
                        "source_split": str(row.split), "vv_file": vv.get(s1), "vh_file": vh.get(s1),
                        "reference_file": refs.get(area), "completeness": not reasons,
                        "validity": "valid" if not reasons else "invalid", "invalid_reasons": reasons,
                        "classification": "IN PASS-3 EXCLUSION SET" if matches else "ELIGIBLE FOR PASS-5",
                        "exclusion_identity_matches": matches})
    excluded = [x for x in samples if x["classification"].startswith("IN ")]
    eligible = [x for x in samples if x["classification"].startswith("ELIGIBLE") and x["validity"] == "valid"]
    invalid = [x for x in samples if x["validity"] != "valid"]
    manifest = {"format_version": 1, "status": "verified", "dataset": "BigEarthNet v2 selected 5000-area subset",
                "archive": str(archive), "archive_sha256": archive_hash, "counts": {"total": len(samples),
                "excluded": len(excluded), "eligible": len(eligible), "invalid": len(invalid)}, "samples": samples}
    _json(output / "dataset_manifest.json", manifest)
    if len(excluded) != 1000 or len(eligible) != 4000 or invalid:
        raise ValueError(f"unexpected eligibility counts: {manifest['counts']}")

    # Balanced within country; fixed 70/15/15 allocation for each 400-area group.
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in eligible: grouped[row["country"]].append(row["area_id"])
    areas = {name: [] for name in ("train", "validation", "test")}
    geography = {}
    for country in sorted(grouped):
        ids = sorted(grouped[country]); rng.shuffle(ids)
        n_train, n_validation = round(.70 * len(ids)), round(.15 * len(ids))
        parts = {"train": ids[:n_train], "validation": ids[n_train:n_train+n_validation], "test": ids[n_train+n_validation:]}
        geography[country] = {key: len(value) for key, value in parts.items()}
        for key in areas: areas[key].extend(parts[key])
    for key in areas: areas[key] = sorted(areas[key])
    sets = {key: set(value) for key, value in areas.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Pass-5 internal split overlap")
    p3_splits = {key: set(pass3["split"]["areas"][key]) for key in ("train", "validation", "test")}
    overlap = {f"pass3_{p3}_vs_pass5_test": sorted(p3_splits[p3] & sets["test"]) for p3 in p3_splits}
    overlap["all_pass3_vs_pass5_test"] = sorted(p3_area & sets["test"])
    if any(overlap.values()):
        raise ValueError(f"Pass-3/Pass-5 leakage: {overlap}")
    split = {"format_version": 1, "seed": seed, "dataset": str(output / "dataset_manifest.json"),
             "excluded_pass3_areas": len(p3_area), "unit": "canonical geographic area",
             "counts": {key: len(value) for key, value in areas.items()}, "geography": geography, "areas": areas}
    _json(output / "split.json", split)
    audit = {"status": "PASS", "identity_keys": ["canonical area ID", "Sentinel-1 identity", "metadata identity"],
             "pass3_count": len(p3_area), "pass5_test_count": len(sets["test"]), "overlap": overlap,
             "pass5_internal_overlaps": {"train_validation": 0, "train_test": 0, "validation_test": 0}}
    _json(output / "leakage_audit.json", audit)
    return manifest, split


def _read_tiff(archive: zipfile.ZipFile, name: str, role: str, expected_dtype: str | None = None):
    raw = archive.read(name)
    try:
        with rasterio.io.MemoryFile(raw) as memory, memory.open() as ds:
            values = ds.read(1)
            bounds, transform = tuple(ds.bounds), tuple(ds.transform)
            if ds.count != 1 or values.shape != (120, 120): raise ValueError(f"{role}: wrong dimensions")
            if ds.crs is None: raise ValueError(f"{role}: invalid CRS")
            if transform[0] <= 0 or transform[4] >= 0 or transform[1] != 0 or transform[3] != 0: raise ValueError(f"{role}: invalid orientation")
            if not np.isfinite(bounds).all() or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]: raise ValueError(f"{role}: invalid bounds")
            if not np.isfinite(values).all(): raise ValueError(f"{role}: NaN/Inf")
            if expected_dtype and str(values.dtype) != expected_dtype: raise ValueError(f"{role}: expected {expected_dtype}, got {values.dtype}")
            meta = {"shape": list(values.shape), "dtype": str(values.dtype), "crs": str(ds.crs),
                    "bounds": list(bounds), "transform": list(transform), "resolution": list(ds.res)}
            return values, meta, hashlib.sha256(raw).hexdigest()
    except rasterio.errors.RasterioIOError as error:
        raise ValueError(f"{role}: corrupted TIFF") from error


def extract_features(archive: Path, manifest: dict, split: dict, output: Path, batch_size: int = 32, device: str | None = None) -> dict:
    started = time.perf_counter(); settings = get_settings()
    selected = set(sum(split["areas"].values(), [])); rows = [x for x in manifest["samples"] if x["area_id"] in selected]
    by_id = {x["area_id"]: x for x in rows}
    ordered_ids = sum((split["areas"][key] for key in ("train", "validation", "test")), [])
    adapter = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device=device)
    s1_gap, physical, targets, validation_rows = [], [], [], []
    representative_tokens = {}
    loading_seconds = extraction_seconds = 0.0
    with zipfile.ZipFile(archive) as outer, tempfile.TemporaryDirectory(prefix="satquery-pass5-") as temp:
        s1_path, ref_path = Path(temp)/"s1.zip", Path(temp)/"reference.zip"
        s1_path.write_bytes(outer.read(f"{ARCHIVE_MEMBER_ROOT}/BigEarthNet-S1-selected.zip"))
        ref_path.write_bytes(outer.read(f"{ARCHIVE_MEMBER_ROOT}/Reference_Maps-selected.zip"))
        with zipfile.ZipFile(s1_path) as s1z, zipfile.ZipFile(ref_path) as refz:
            for offset in range(0, len(ordered_ids), batch_size):
                batch_ids = ordered_ids[offset:offset+batch_size]; batch_sar=[]
                t0=time.perf_counter()
                for area in batch_ids:
                    row=by_id[area]
                    vv, vm, vhash = _read_tiff(s1z, row["vv_file"], f"{area} VV", "float32")
                    vh, hm, hhash = _read_tiff(s1z, row["vh_file"], f"{area} VH", "float32")
                    ref, rm, rhash = _read_tiff(refz, row["reference_file"], f"{area} reference", "uint16")
                    if (vm["crs"], vm["bounds"], vm["transform"]) != (hm["crs"], hm["bounds"], hm["transform"]): raise ValueError(f"{area}: VV/VH grid mismatch")
                    if (vm["crs"], vm["bounds"], vm["transform"]) != (rm["crs"], rm["bounds"], rm["transform"]): raise ValueError(f"{area}: SAR/reference mismatch")
                    sar=np.stack((vv,vh)).astype(np.float32); batch_sar.append(sar)
                    physical.append(extract_physical({"sar":sar})[0])
                    target_contract=patch_label_targets(torch.from_numpy(ref[None]))
                    counts=target_contract["class_counts"].sum(1).squeeze(0).double().numpy()
                    if counts.sum() <= 0: raise ValueError(f"{area}: no labelled target pixels")
                    targets.append((counts/counts.sum()).astype(np.float32))
                    validation_rows.append({"area_id":area,"sar_shape":[2,120,120],"sar_band_order":["VV","VH"],
                        "vv":vm,"vh":hm,"reference":rm,"hashes":{"vv":vhash,"vh":hhash,"reference":rhash},
                        "target_shape":[19],"target_sum":float(targets[-1].sum())})
                loading_seconds += time.perf_counter()-t0
                raw=torch.from_numpy(np.asarray(batch_sar)).float().to(adapter.device)
                mean=raw.mean((-2,-1),keepdim=True); std=raw.std((-2,-1),keepdim=True)
                normalized=torch.where(std>0, ((raw-(mean-2*std))/(4*std)).clamp(0,1), torch.zeros_like(raw))
                t0=time.perf_counter()
                with torch.inference_mode():
                    tokens=adapter.model.s1_encoder(imgs=normalized,attn_bias=adapter.model.attn_bias.to(adapter.device))
                    gap=adapter.model.GAP_FFN_s1(tokens.mean(dim=1))
                extraction_seconds += time.perf_counter()-t0
                if tuple(tokens.shape)!=(len(batch_ids),225,768) or tuple(gap.shape)!=(len(batch_ids),768) or not torch.isfinite(tokens).all() or not torch.isfinite(gap).all():
                    raise ValueError("invalid SAR CROMA representation")
                s1_gap.extend(gap.cpu().numpy())
                for i, area in enumerate(batch_ids):
                    if len(representative_tokens)<3: representative_tokens[area]=tokens[i].cpu().numpy()
                print(f"Pass 5 representation {min(offset+batch_size,len(ordered_ids))}/{len(ordered_ids)}", flush=True)
    arrays={"sar_croma_gap":np.asarray(s1_gap,dtype=np.float32),
            "sar_physical":np.asarray(physical,dtype=np.float32),"targets":np.asarray(targets,dtype=np.float32)}
    _json(output/"artifact_identities.json",{"sample_id":"pass5-eligible-4000","area_ids":ordered_ids})
    receipt=save_artifact_bundle(output/"artifacts",sample_id="pass5-eligible-4000",arrays=arrays,
        provenance={"archive":str(archive),"archive_sha256":manifest["archive_sha256"],"split_sha256":_sha256(output/"split.json"),
        "croma_source":str(settings.croma_source),"croma_checkpoint":str(settings.croma_checkpoint),
        "croma_checkpoint_sha256":_sha256(settings.croma_checkpoint),"normalization":"official per-channel mean +/- 2 sample std clipped to [0,1]",
        "target":"Pass-3 scene-level 19-class labelled-pixel coverage"})
    verified=verify_artifact_bundle(output/"artifacts")
    for area,tokens in representative_tokens.items():
        save_artifact_bundle(output/"representative_tokens"/area,sample_id=area,arrays={"sar_encodings":tokens},provenance=receipt["provenance"])
        verify_artifact_bundle(output/"representative_tokens"/area)
    report={"status":"validated","areas":len(rows),"loading_seconds":loading_seconds,"croma_extraction_seconds":extraction_seconds,
            "total_seconds":time.perf_counter()-started,"device":str(adapter.device),"croma_shapes":{"tokens":[len(rows),225,768],"scene":[len(rows),768]},
            "physical_shape":list(arrays["sar_physical"].shape),"target_shape":list(arrays["targets"].shape),
            "artifact_receipt_status":verified["status"],"representative_validation":validation_rows[:10]}
    _json(output/"data_validation.json",report); return report


def train_and_evaluate(output: Path, split: dict, seed: int=53, epochs: int=80, patience: int=12, device: str | None=None) -> dict:
    started=time.perf_counter(); receipt=verify_artifact_bundle(output/"artifacts")
    root=output/"artifacts"; ids=np.asarray(json.loads((output/"artifact_identities.json").read_text(encoding="utf-8"))["area_ids"]); y=np.load(root/"targets.npy",allow_pickle=False)
    features={"sar_croma":np.load(root/"sar_croma_gap.npy",allow_pickle=False),"sar_physical":np.load(root/"sar_physical.npy",allow_pickle=False)}
    split_by_id={area:key for key,values in split["areas"].items() for area in values}; labels=np.asarray([split_by_id[x] for x in ids])
    masks={key:labels==key for key in ("train","validation","test")}; test_ids=ids[masks["test"]]; test_truth=y[masks["test"]]
    predictions={}; metrics={}; logs={}; checkpoints={}; training_seconds={}
    train_mean=y[masks["train"]].mean(0); train_mean/=train_mean.sum(); predictions["constant"]=np.repeat(train_mean[None],len(test_truth),axis=0)
    metrics["constant"]=coverage_metrics(predictions["constant"],test_truth,test_ids,bootstrap_repeats=2000,seed=seed)
    validation_predictions=np.repeat(train_mean[None],int(masks["validation"].sum()),axis=0)
    validation_metrics={"constant":coverage_metrics(validation_predictions,y[masks["validation"]],ids[masks["validation"]],bootstrap_repeats=500,seed=seed)}
    for name,x in features.items():
        parts={key:ProbeSplit(key,torch.from_numpy(x[masks[key]]),torch.from_numpy(y[masks[key]]),ids[masks[key]]) for key in masks}
        t0=time.perf_counter(); model,log=fit_probe(parts["train"],parts["validation"],seed=seed,max_epochs=epochs,patience=patience,batch_size=512,device=device or ("cuda" if torch.cuda.is_available() else "cpu")); training_seconds[name]=time.perf_counter()-t0
        predictions[name]=predict_rows(model,parts["test"].x).numpy(); val_pred=predict_rows(model,parts["validation"].x).numpy()
        metrics[name]=coverage_metrics(predictions[name],test_truth,test_ids,bootstrap_repeats=2000,seed=seed)
        validation_metrics[name]=coverage_metrics(val_pred,y[masks["validation"]],ids[masks["validation"]],bootstrap_repeats=500,seed=seed)
        model_root=output/name; model_root.mkdir(parents=True,exist_ok=True); checkpoint=model_root/"checkpoint.pt"
        torch.save({"state_dict":model.state_dict(),"feature":name,"input_dimension":int(x.shape[1]),"seed":seed,"selected_epoch":log["selected_epoch"],"split":str(output/"split.json")},checkpoint)
        _json(model_root/"config.json",{"feature":name,"dimension":int(x.shape[1]),"seed":seed,"epochs":epochs,"patience":patience,"batch_size":512,"selection":"validation_soft_target_cross_entropy"})
        _json(model_root/"training_log.json",log); _json(model_root/"validation_metrics.json",validation_metrics[name]); checkpoints[name]={"path":str(checkpoint),"sha256":_sha256(checkpoint)}; logs[name]=log
    independent={name:_independent(value,test_truth) for name,value in predictions.items()}
    difference=max(abs(independent[name][metric]-metrics[name][metric]) for name in predictions for metric in ("mae_pp","rmse_pp","bias_pp","dominant_class_accuracy") if independent[name][metric] is not None)
    if difference>1e-12: raise ValueError(f"independent metric mismatch: {difference}")
    pred_payload={"area_ids":test_ids.tolist(),"truth":test_truth.tolist(),"models":{k:v.tolist() for k,v in predictions.items()}}
    _json(output/"comparison"/"predictions.json",pred_payload); _json(output/"comparison"/"metrics.json",metrics)
    # Reload CROMA checkpoint and reproduce all test predictions exactly/tolerantly.
    from .phase3_6_benchmark import ProbeHead
    ck=torch.load(checkpoints["sar_croma"]["path"],map_location="cpu",weights_only=True); reloaded=ProbeHead(ck["input_dimension"]); reloaded.load_state_dict(ck["state_dict"]); reloaded.eval()
    repeat=predict_rows(reloaded,torch.from_numpy(features["sar_croma"][masks["test"]])).numpy(); repro_diff=float(np.max(np.abs(repeat-predictions["sar_croma"])))
    if repro_diff>1e-7: raise ValueError(f"checkpoint reproducibility mismatch: {repro_diff}")
    countries={x["area_id"]:x["country"] for x in json.loads((output/"dataset_manifest.json").read_text(encoding="utf-8"))["samples"]}
    geographic={}
    for country in sorted({countries[x] for x in test_ids}):
        mask=np.asarray([countries[x]==country for x in test_ids]); geographic[country]={name:_independent(value[mask],test_truth[mask]) for name,value in predictions.items()}
    provenance={"status":"complete","artifact_receipt":receipt,"split_sha256":_sha256(output/"split.json"),"checkpoints":checkpoints,
        "training_seconds":training_seconds,"test_used_for_selection":False,"independent_metric_max_abs_difference":difference,
        "reproducibility":{"status":"PASS","max_abs_prediction_difference":repro_diff,"tolerance":1e-7},"geographic_metrics":geographic,
        "runtime_seconds":time.perf_counter()-started,"system":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}}
    _json(output/"comparison"/"provenance.json",provenance); return {"metrics":metrics,"provenance":provenance}


def _independent(predictions: np.ndarray, truth: np.ndarray) -> dict:
    p,y=np.asarray(predictions,dtype=np.float64),np.asarray(truth,dtype=np.float64); error=(p-y)*100
    actual,predicted=y.argmax(1),p.argmax(1); untied=(y==y.max(1,keepdims=True)).sum(1)==1
    return {"mae_pp":float(np.abs(error).mean()),"rmse_pp":float(np.sqrt(np.square(error).mean())),"bias_pp":float(error.mean()),
            "dominant_class_accuracy":float((actual[untied]==predicted[untied]).mean()) if untied.any() else None,
            "per_class_mae_pp":np.abs(error).mean(0).tolist(),"per_class_rmse_pp":np.sqrt(np.square(error).mean(0)).tolist(),"per_class_bias_pp":error.mean(0).tolist()}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--archive",type=Path,required=True); parser.add_argument("--output",type=Path,default=Path("experiments/pass5")); parser.add_argument("--pass3-manifest",type=Path,default=Path("experiments/pass3/dataset_manifest.json")); parser.add_argument("--seed",type=int,default=53); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--device",default=None); parser.add_argument("--epochs",type=int,default=80); parser.add_argument("--patience",type=int,default=12); parser.add_argument("--gates-only",action="store_true"); args=parser.parse_args()
    manifest,split=build_gates(args.archive,args.pass3_manifest,args.output,args.seed)
    if args.gates_only: print(json.dumps({"status":"gates_passed","counts":manifest["counts"],"split":split["counts"]},indent=2)); return
    extract_features(args.archive,manifest,split,args.output,args.batch_size,args.device)
    result=train_and_evaluate(args.output,split,args.seed,args.epochs,args.patience,args.device)
    print(json.dumps({"status":"complete","counts":manifest["counts"],"split":split["counts"],"metrics":{k:{m:v[m] for m in ("mae_pp","rmse_pp","bias_pp","dominant_class_accuracy")} for k,v in result["metrics"].items()},"independent_difference":result["provenance"]["independent_metric_max_abs_difference"],"reproducibility":result["provenance"]["reproducibility"]},indent=2))


if __name__ == "__main__": main()
