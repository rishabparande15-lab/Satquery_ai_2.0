"""Fresh raw-area scene-coverage smoke experiment.

The task is deliberately scene-level: each area's labelled reference pixels are
aggregated to one 19-class coverage distribution. It does not claim token-level
grounding or production model readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import get_settings
from .croma_adapter import CROMAAdapter
from .dataset_loader import OPTICAL_BANDS, SAR_BANDS, discover_samples, load_sample
from .gee_features import LocalRasterFeatureProvider
from .phase1_foundation import CLASSES, patch_label_targets, coverage_metrics, validate_croma_features
from .phase3_6_benchmark import ProbeSplit, fit_probe, predict_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _scene_target(reference: np.ndarray) -> np.ndarray:
    target = patch_label_targets(torch.from_numpy(reference[None]))
    counts = target["class_counts"].sum((0, 1)).numpy().astype(np.float32)
    if counts.sum() <= 0: raise ValueError("Scene has no eligible labelled reference pixels")
    return counts / counts.sum()


def run(dataset_root: Path, output_root: Path, *, per_split: int = 2, seed: int = 17,
        epochs: int = 8, device: str | None = None) -> dict:
    if per_split < 1 or per_split * 3 < 5 or per_split * 3 > 10:
        raise ValueError("Smoke run must contain 5-10 areas with every split represented")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    started = time.perf_counter(); output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(Path(dataset_root), strict=True)
    import pandas as pd
    frame = pd.read_parquet(Path(dataset_root) / "metadata.parquet", columns=["patch_id", "split", "country"])
    split_by_id = dict(zip(frame.patch_id.astype(str), frame.split.astype(str)))
    country_by_id = dict(zip(frame.patch_id.astype(str), frame.country.astype(str)))
    selected = {name: [s for s in samples if split_by_id.get(s.patch_id) == name][:per_split]
                for name in ("train", "validation", "test")}
    if any(len(items) != per_split for items in selected.values()): raise ValueError("Insufficient areas in one or more splits")
    settings = get_settings(); adapter = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device=device)
    provider = LocalRasterFeatureProvider(); rows=[]; representations={name: [] for name in ("physical", "optical", "sar", "joint", "hybrid")}
    targets=[]; splits=[]; ids=[]
    for split, items in selected.items():
        for sample in items:
            prepared = load_sample(sample)
            deep = {key: value.detach().cpu().to(torch.float32) for key, value in adapter.infer(prepared.raw_optical, prepared.raw_sar).items()}
            validate_croma_features(deep, 1)
            physical, report = provider.extract(prepared.raw_optical, prepared.raw_sar, list(OPTICAL_BANDS), list(SAR_BANDS))
            target = _scene_target(prepared.reference)
            vectors = {"physical": physical.astype(np.float32), "optical": deep["optical_GAP"][0].numpy(),
                       "sar": deep["SAR_GAP"][0].numpy(), "joint": deep["joint_GAP"][0].numpy()}
            vectors["hybrid"] = np.concatenate((vectors["physical"], vectors["joint"])).astype(np.float32)
            for name, value in vectors.items(): representations[name].append(value)
            targets.append(target); splits.append(split); ids.append(sample.patch_id)
            rows.append({"area_id": sample.patch_id, "split": split, "country": country_by_id[sample.patch_id],
                         "raw_shapes": {"optical": list(prepared.raw_optical.shape), "sar": list(prepared.raw_sar.shape), "reference": list(prepared.reference.shape)},
                         "croma_shapes": {key: list(value.shape) for key, value in deep.items()}, "physical_dimension": int(len(physical)),
                         "reference_sha256": _sha256(sample.reference_map)})
    y = np.stack(targets).astype(np.float32); ids_array=np.asarray(ids); split_array=np.asarray(splits)
    results={}; predictions={}
    train_mean = y[split_array == "train"].mean(0); train_mean /= train_mean.sum()
    constant = np.repeat(train_mean[None], int((split_array == "test").sum()), axis=0)
    results["constant"] = coverage_metrics(constant, y[split_array == "test"], ids_array[split_array == "test"], bootstrap_repeats=50, seed=seed)
    predictions["constant"] = constant.tolist()
    checkpoint_dir=output_root/"checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    logs={}
    for name, values in representations.items():
        x=np.stack(values).astype(np.float32)
        parts={part: ProbeSplit(part, torch.from_numpy(x[split_array == part]), torch.from_numpy(y[split_array == part]), ids_array[split_array == part]) for part in ("train","validation","test")}
        before={key:value.detach().clone() for key,value in torch.nn.Linear(x.shape[1],19).state_dict().items()}
        model, log=fit_probe(parts["train"],parts["validation"],seed=seed,max_epochs=epochs,patience=epochs,batch_size=16,device=str(adapter.device))
        pred=predict_rows(model,parts["test"].x).numpy(); predictions[name]=pred.tolist()
        results[name]=coverage_metrics(pred,parts["test"].y.numpy(),parts["test"].area_ids,bootstrap_repeats=50,seed=seed)
        torch.save({"state_dict":model.state_dict(),"feature":name,"input_dimension":x.shape[1],"selected_epoch":log["selected_epoch"],"seed":seed},checkpoint_dir/f"{name}.pt")
        logs[name]=log
    config={"dataset_root":str(Path(dataset_root)),"per_split":per_split,"areas":len(ids),"seed":seed,"epochs":epochs,
            "device":str(adapter.device),"task":"scene-level 19-class labelled-pixel coverage", "class_order":[x[0] for x in CLASSES],
            "target":"sum class pixels over all 225 blocks, divide by total labelled class pixels; excluded pixels omitted"}
    payload={"status":"complete","timestamp":datetime.now(timezone.utc).isoformat(),"config":config,"samples":rows,"metrics":results,
             "training":logs,"test_area_ids":ids_array[split_array=="test"].tolist(),"test_truth":y[split_array=="test"].tolist(),"predictions":predictions,
             "leakage":{"area_overlap":False,"split_unit":"whole area","test_used_for_selection":False},"runtime_seconds":time.perf_counter()-started}
    for name,value in (("config.json",config),("split.json",rows),("metrics.json",results),("predictions.json",{"area_ids":payload["test_area_ids"],"truth":payload["test_truth"],"models":predictions}),("logs.json",logs),("provenance.json",payload)):
        (output_root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return payload


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-root",type=Path,required=True); parser.add_argument("--output-root",type=Path,default=Path("experiments/smoke")); parser.add_argument("--per-split",type=int,default=2); parser.add_argument("--seed",type=int,default=17); parser.add_argument("--epochs",type=int,default=8); parser.add_argument("--device",default=None)
    args=parser.parse_args(); result=run(args.dataset_root,args.output_root,per_split=args.per_split,seed=args.seed,epochs=args.epochs,device=args.device)
    print(json.dumps({"status":result["status"],"areas":result["config"]["areas"],"runtime_seconds":result["runtime_seconds"],"metrics":{k:{"mae_pp":v["mae_pp"],"rmse_pp":v["rmse_pp"],"bias_pp":v["bias_pp"],"dominant_class_accuracy":v["dominant_class_accuracy"]} for k,v in result["metrics"].items()}},indent=2))


if __name__ == "__main__": main()
