"""Shared-data Phase 3.6 representation benchmark utilities.

Generated data and reports belong under ignored experiments/comparison/. This
module never downloads data and never changes the production model path.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .gee_features import LocalRasterFeatureProvider
from .hybrid_fusion import HybridFusion
from .phase1_foundation import CLASSES, OPTICAL_BANDS, SAR_BANDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ProbeSplit:
    split: str
    x: torch.Tensor
    y: torch.Tensor
    area_ids: np.ndarray


class ProbeHead(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, 19)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

    @torch.inference_mode()
    def predict(self, value: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(value).softmax(-1)


def coverage_loss(logits: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    if logits.shape != fractions.shape or logits.shape[-1] != 19:
        raise ValueError("Expected aligned 19-class logits and fractions")
    if not torch.isfinite(logits).all() or not torch.isfinite(fractions).all():
        raise ValueError("Probe inputs must be finite")
    if (fractions < 0).any() or not torch.allclose(fractions.sum(-1), torch.ones_like(fractions[..., 0]), atol=1e-6, rtol=0):
        raise ValueError("Targets must be complete coverage distributions")
    return -(fractions * logits.log_softmax(-1)).sum(-1).mean()


def fit_probe(train: ProbeSplit, validation: ProbeSplit, *, seed: int = 17,
              max_epochs: int = 60, patience: int = 10, learning_rate: float = .001,
              batch_size: int = 1024, device: str = "cpu", progress=None):
    """Dimension-general form of Ravi's validation-selected linear-head protocol."""
    if train.split != "train" or validation.split != "validation":
        raise ValueError("Test data cannot select a probe")
    if set(train.area_ids) & set(validation.area_ids):
        raise ValueError("Training and validation area identities overlap")
    if train.x.ndim != 2 or validation.x.ndim != 2 or train.x.shape[1] != validation.x.shape[1]:
        raise ValueError("Probe feature dimensions are invalid")
    for part in (train, validation):
        if part.y.shape != (len(part.x), 19) or part.x.dtype != torch.float32 or part.y.dtype != torch.float32:
            raise ValueError("Invalid probe rows")
        coverage_loss(torch.zeros_like(part.y), part.y)
    mean = train.x.mean(0)
    scale = train.x.std(0, correction=0)
    scale = torch.where(scale < 1e-5, torch.ones_like(scale), scale)
    x, y = ((train.x - mean) / scale).to(device), train.y.to(device)
    vx, vy = ((validation.x - mean) / scale).to(device), validation.y.to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ProbeHead(train.x.shape[1]).to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        model.train(); order = torch.randperm(len(x), generator=generator).to(device); total = 0.0
        for start in range(0, len(x), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = coverage_loss(model(x[idx]), y[idx]); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval(); val_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(vx), batch_size):
                truth = vy[start:start + batch_size]
                val_sum += float(coverage_loss(model(vx[start:start + batch_size]), truth)) * len(truth)
        val_loss = val_sum / len(vx)
        history.append({"epoch": epoch, "train_loss": total / len(x), "validation_loss": val_loss})
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        if progress: progress(f"Epoch {epoch}: train {total/len(x):.4f}, validation {val_loss:.4f}; best {best_epoch}")
        if epoch - best_epoch >= patience: break
    model.load_state_dict(best_state); model = model.cpu().eval()
    with torch.inference_mode():
        before = model.predict((validation.x[:256] - mean) / scale)
        model.linear.weight.div_(scale); model.linear.bias.sub_(model.linear.weight @ mean)
        after = model.predict(validation.x[:256]); discrepancy = float((before-after).abs().max())
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)
    return model, {"selected_epoch": best_epoch, "best_validation_loss": best, "history": history,
        "selection_metric": "validation_soft_target_cross_entropy", "test_used_for_selection": False,
        "max_epochs": max_epochs, "patience": patience, "seed": seed, "learning_rate": learning_rate,
        "batch_size": batch_size, "optimizer": "AdamW", "weight_decay": .01,
        "feature_standardization": "train-only mean/std folded into linear probe",
        "folded_normalization_max_abs_error": discrepancy}


@torch.inference_mode()
def predict_rows(model, x, batch_size=2048):
    return torch.cat([model.predict(x[i:i+batch_size]).cpu() for i in range(0, len(x), batch_size)])


def assert_no_leakage(samples: list[dict], eligible_ids: dict[str, list[str]]) -> dict:
    keys=("patch_id","s1_name")
    evidence={}
    for key in keys:
        by_split={s:{row[key] for row in samples if row["split"]==s} for s in ("train","validation","test")}
        overlap={f"{a}_{b}":sorted(by_split[a]&by_split[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
        evidence[key]=overlap
        if any(values for values in overlap.values()): raise ValueError(f"{key} leakage")
    if any(set(eligible_ids[a]) & set(eligible_ids[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Eligible target identity leakage")
    return {"status":"PASS","checks":evidence,"test_used_for_training":False,"test_used_for_epoch_selection":False}


def _batch_dir(root: Path, entry: dict) -> Path:
    return root if entry["directory"] == "." else root / entry["directory"]


def build_hybrid_splits(pipeline_root: Path, *, device: str = "cuda", seed: int = 42, progress=None):
    prepared, features, targets = (pipeline_root/name for name in ("prepared","features","targets"))
    pm, fm, tm = map(lambda p: load_json(p/"manifest.json"), (prepared,features,targets))
    if not (pm["sample_count"] == fm["sample_count"] == tm["sample_count"] == 1000): raise ValueError("Expected the shared 1,000-area manifests")
    torch.manual_seed(seed)
    fusion = HybridFusion(62,2304).to(device).eval()
    provider=LocalRasterFeatureProvider(); parts={s:{"x":[],"y":[],"ids":[]} for s in ("train","validation","test")}; samples=[]
    physical_names=None; area_vectors={}; started=time.perf_counter()
    for number,(pb,fb,tb) in enumerate(zip(pm["batches"],fm["batches"],tm["batches"],strict=True)):
        pdir,fdir,tdir=_batch_dir(prepared,pb),_batch_dir(features,fb),_batch_dir(targets,tb)
        meta=load_json(pdir/"batch.json"); raw=torch.load(pdir/"raw_inputs.pt",map_location="cpu",weights_only=True)
        deep=torch.load(fdir/"features.pt",map_location="cpu",weights_only=True); target=torch.load(tdir/"targets.pt",map_location="cpu",weights_only=True)
        if len(meta["samples"])!=pb["sample_count"] or target["class_fractions"].shape[:2]!=(pb["sample_count"],225): raise ValueError("Batch identity/target mismatch")
        physical=[]
        for i,sample in enumerate(meta["samples"]):
            vector,report=provider.extract(raw["optical_images"][i].numpy(),raw["SAR_images"][i].numpy(),list(OPTICAL_BANDS),list(SAR_BANDS))
            names=tuple(report["feature_names"]); physical_names=physical_names or names
            if names!=physical_names or len(vector)!=62: raise ValueError("Physical feature schema changed")
            physical.append(vector); samples.append(sample)
        physical=np.asarray(physical,dtype=np.float32)
        pooled=torch.cat((deep["optical_GAP"],deep["SAR_GAP"],deep["joint_GAP"]),-1).to(torch.float32)
        with torch.inference_mode(): area=fusion(torch.from_numpy(physical).to(device),pooled.to(device)).cpu()
        for i,sample in enumerate(meta["samples"]):
            mask=target["fully_labeled_mask"][i]; split=sample["split"]; n=int(mask.sum())
            parts[split]["x"].append(area[i].expand(n,-1)); parts[split]["y"].append(target["class_fractions"][i,mask]); parts[split]["ids"].extend([sample["patch_id"]]*n); area_vectors[sample["patch_id"]]=area[i].numpy()
        if progress: progress(f"Built SatQuery representations for {sum(len(v) for v in area_vectors.values()) if False else len(area_vectors)}/1000 areas")
    splits={s:ProbeSplit(s,torch.cat(v["x"]),torch.cat(v["y"]),np.asarray(v["ids"])) for s,v in parts.items()}
    return splits,samples,area_vectors,{"physical_dimension":62,"pooled_croma_dimension":2304,"hybrid_dimension":192,"hybrid_seed":seed,"fusion_weights":"deterministic seeded untrained existing HybridFusion","physical_feature_names":list(physical_names),"runtime_seconds":time.perf_counter()-started,"prepared_manifest_sha256":sha256(prepared/"manifest.json"),"feature_manifest_sha256":sha256(features/"manifest.json"),"target_manifest_sha256":sha256(targets/"manifest.json")}


def dataset_manifest(pipeline_root: Path) -> dict:
    prepared=pipeline_root/"prepared"; manifest=load_json(prepared/"manifest.json"); samples=[]
    for entry in manifest["batches"]:
        for item in load_json(_batch_dir(prepared,entry)/"batch.json")["samples"]:
            sources=item["sources"]
            reference=next(x for x in sources if x["path"].endswith("_reference_map.tif"))
            optical=[x for x in sources if "BigEarthNet-S2" in x["path"]]
            sar=[x for x in sources if "BigEarthNet-S1" in x["path"]]
            timestamps=re.findall(r"20\d{6}T\d{6}", item["patch_id"]+" "+item["s1_name"])
            samples.append({"sample_id":item["patch_id"],"split":item["split"],"country":item["country"],
                "sentinel1_identity":item["s1_name"],"sentinel2_identity":item["patch_id"],
                "reference_map_identity":reference["path"],"source_files":{"optical":optical,"sar":sar,"reference_map":reference},
                "crs":item["crs"],"bounds":item["bounds"],"transform":item["transform"],"resolution":[10.0,10.0],
                "acquisition_metadata":{"sentinel2_timestamp":timestamps[0] if timestamps else None,"sentinel1_timestamp":timestamps[1] if len(timestamps)>1 else None}})
    return {"format_version":1,"dataset":"BigEarthNet v2 pinned reference subset-1000","sample_count":len(samples),"split_counts":{s:sum(x["split"]==s for x in samples) for s in ("train","validation","test")},"samples":samples,"prepared_manifest_sha256":sha256(prepared/"manifest.json"),"normalization_profile":manifest["normalization_profile"],"croma_source_revision":manifest["croma_source_revision"]}


def shared_evaluation_manifest(pipeline_root: Path) -> dict:
    prepared,targets=pipeline_root/"prepared",pipeline_root/"targets"
    pm,tm=load_json(prepared/"manifest.json"),load_json(targets/"manifest.json"); rows=[]
    for pb,tb in zip(pm["batches"],tm["batches"],strict=True):
        samples=load_json(_batch_dir(prepared,pb)/"batch.json")["samples"]
        values=torch.load(_batch_dir(targets,tb)/"targets.pt",map_location="cpu",weights_only=True)
        for index,sample in enumerate(samples):
            if sample["split"]=="test": rows.append({"sample_id":sample["patch_id"],"eligible_block_ids":torch.where(values["fully_labeled_mask"][index])[0].tolist()})
    return {"format_version":1,"test_area_ids":[x["sample_id"] for x in rows],"eligible_blocks":rows,
        "eligible_block_count":sum(len(x["eligible_block_ids"]) for x in rows),"class_order":[x[0] for x in CLASSES],
        "exclusion_rule":"fully_labeled_mask only; unknown/excluded pixels are not redistributed","target_manifest_sha256":sha256(targets/"manifest.json")}


def write_per_class(path: Path, metrics: dict):
    rows=metrics["classes"]
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["index","name","mae_pp","rmse_pp","bias_pp","present_mae_pp","present_tokens","present_areas","dominant_precision","dominant_recall","dominant_f1"])
        writer.writeheader(); writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in rows)


def environment() -> dict:
    return {"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
