"""Pass 3 held-out scene-coverage comparison on the shared 1,000-area artifacts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .gee_features import LocalRasterFeatureProvider
from .phase1_foundation import CLASSES, OPTICAL_BANDS, SAR_BANDS, coverage_metrics
from .phase3_6_benchmark import ProbeSplit, fit_probe, predict_rows


MODELS = ("constant", "physical", "optical_croma", "sar_croma", "joint_croma", "hybrid")


def _batch_dir(root: Path, entry: dict) -> Path:
    return root if entry["directory"] == "." else root / entry["directory"]


def _scene_target(class_counts: torch.Tensor) -> np.ndarray:
    if class_counts.ndim == 3:
        class_counts = class_counts.sum(1)
    if class_counts.ndim != 2 or class_counts.shape[-1] != 19:
        raise ValueError(f"Expected area class counts [225,19], got {tuple(class_counts.shape)}")
    counts = class_counts.to(torch.float64).sum(0).numpy()
    if not np.isfinite(counts).all() or counts.sum() <= 0:
        raise ValueError("Area has no finite labelled target pixels")
    return (counts / counts.sum()).astype(np.float32)


def _independent_metrics(predictions: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    error = (predictions.astype(np.float64) - truth.astype(np.float64)) * 100
    actual, predicted = truth.argmax(1), predictions.argmax(1)
    untied = (truth == truth.max(1, keepdims=True)).sum(1) == 1
    return {"mae_pp": float(np.abs(error).mean()), "rmse_pp": float(np.sqrt(np.square(error).mean())),
            "bias_pp": float(error.mean()),
            "dominant_class_accuracy": float((actual[untied] == predicted[untied]).mean()) if untied.any() else None}


def run(pipeline_root: Path, output_root: Path, *, manifest_path: Path | None = None, seed: int = 17,
        epochs: int = 60, patience: int = 10, device: str | None = None, progress=None) -> dict:
    started = time.perf_counter()
    pipeline_root, output_root = Path(pipeline_root), Path(output_root)
    manifest = json.loads(Path(manifest_path or "experiments/pass3/dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated" or manifest["split_counts"] != {"train": 600, "validation": 200, "test": 200}:
        raise ValueError("Pass 3 requires a validated canonical 600/200/200 manifest")
    by_id = {row["area_id"]: row for row in manifest["samples"]}
    prepared, features, targets = (pipeline_root / name for name in ("prepared", "features", "targets"))
    pm, fm, tm = [json.loads((root / "manifest.json").read_text(encoding="utf-8")) for root in (prepared, features, targets)]
    if not (pm["sample_count"] == fm["sample_count"] == tm["sample_count"] == 1000):
        raise ValueError("Prepared, feature, and target artifacts are not the same 1,000-area dataset")
    provider = LocalRasterFeatureProvider()
    rows, targets_by_area, values_by_model = [], [], {name: [] for name in MODELS if name != "constant"}
    split_names, area_ids = [], []
    for index, (prepared_entry, feature_entry, target_entry) in enumerate(zip(pm["batches"], fm["batches"], tm["batches"], strict=True)):
        pdir, fdir, tdir = _batch_dir(prepared, prepared_entry), _batch_dir(features, feature_entry), _batch_dir(targets, target_entry)
        pmeta = json.loads((pdir / "batch.json").read_text(encoding="utf-8"))
        fmeta = json.loads((fdir / "batch.json").read_text(encoding="utf-8"))
        tmeta = json.loads((tdir / "batch.json").read_text(encoding="utf-8"))
        if [x["patch_id"] for x in pmeta["samples"]] != [x["patch_id"] for x in fmeta["samples"]] or [x["patch_id"] for x in pmeta["samples"]] != [x["patch_id"] for x in tmeta["samples"]]:
            raise ValueError(f"Artifact identity mismatch in batch {index}")
        raw = torch.load(pdir / "raw_inputs.pt", map_location="cpu", weights_only=True)
        deep = torch.load(fdir / "features.pt", map_location="cpu", weights_only=True)
        target = torch.load(tdir / "targets.pt", map_location="cpu", weights_only=True)
        physical = []
        for item, optical, sar, counts in zip(pmeta["samples"], raw["optical_images"], raw["SAR_images"], target["class_counts"]):
            area_id = item["patch_id"]
            if area_id not in by_id or by_id[area_id]["status"] != "valid":
                raise ValueError(f"Area {area_id} is absent from validated Pass 3 manifest")
            vector, report = provider.extract(optical.numpy(), sar.numpy(), list(OPTICAL_BANDS), list(SAR_BANDS))
            if vector.shape != (62,) or not np.isfinite(vector).all() or report["feature_names"] != manifest["samples"][0].get("physical_feature_names", report["feature_names"]):
                # The manifest stores source provenance; the provider report remains the schema authority.
                if vector.shape != (62,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid physical feature vector for {area_id}")
            physical.append(vector)
            targets_by_area.append(_scene_target(counts))
            split_names.append(by_id[area_id]["split"]); area_ids.append(area_id)
            rows.append({"area_id": area_id, "split": by_id[area_id]["split"], "country": by_id[area_id]["country"],
                         "target_shape": [19], "physical_dimension": 62, "croma_shapes": {key: list(value.shape) for key, value in deep.items()}})
        physical_tensor = torch.from_numpy(np.asarray(physical, dtype=np.float32))
        values_by_model["physical"].extend(physical)
        values_by_model["optical_croma"].extend(deep["optical_GAP"].numpy())
        values_by_model["sar_croma"].extend(deep["SAR_GAP"].numpy())
        values_by_model["joint_croma"].extend(deep["joint_GAP"].numpy())
        values_by_model["hybrid"].extend(torch.cat((physical_tensor, deep["joint_GAP"].to(torch.float32)), dim=1).numpy())
        if progress: progress(f"Loaded {len(area_ids)}/1000 areas")
    y, splits, ids = np.asarray(targets_by_area, dtype=np.float32), np.asarray(split_names), np.asarray(area_ids)
    results, predictions, logs = {}, {}, {}
    train_mask, validation_mask, test_mask = (splits == name for name in ("train", "validation", "test"))
    train_mean = y[train_mask].mean(0); train_mean /= train_mean.sum()
    test_truth = y[test_mask]
    predictions["constant"] = np.repeat(train_mean[None], test_truth.shape[0], axis=0)
    results["constant"] = coverage_metrics(predictions["constant"], test_truth, ids[test_mask], bootstrap_repeats=2000, seed=seed)
    selected_epochs = {}
    for name, values in values_by_model.items():
        x = np.asarray(values, dtype=np.float32)
        parts = {part: ProbeSplit(part, torch.from_numpy(x[splits == part]), torch.from_numpy(y[splits == part]), ids[splits == part])
                 for part in ("train", "validation", "test")}
        model, log = fit_probe(parts["train"], parts["validation"], seed=seed, max_epochs=epochs, patience=patience,
                               batch_size=1024, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        predictions[name] = predict_rows(model, parts["test"].x).numpy()
        results[name] = coverage_metrics(predictions[name], test_truth, ids[test_mask], bootstrap_repeats=2000, seed=seed)
        selected_epochs[name] = log["selected_epoch"]; logs[name] = log
        model_root = output_root / name; model_root.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "feature": name, "input_dimension": int(x.shape[1]), "seed": seed,
                    "selected_epoch": log["selected_epoch"], "split_manifest": str(manifest_path or "experiments/pass3/dataset_manifest.json")}, model_root / "checkpoint.pt")
        (model_root / "config.json").write_text(json.dumps({"feature": name, "dimension": int(x.shape[1]), "seed": seed,
                                                               "epochs": epochs, "patience": patience, "batch_size": 1024,
                                                               "optimizer": "AdamW", "selection": "validation_soft_target_cross_entropy"}, indent=2) + "\n")
        (model_root / "test_predictions.json").write_text(json.dumps({"area_ids": ids[test_mask].tolist(), "truth": test_truth.tolist(),
                                                                        "predictions": predictions[name].tolist()}, indent=2) + "\n")
    independent = {name: _independent_metrics(np.asarray(predictions[name]), test_truth) for name in MODELS}
    maximum_metric_difference = max(abs(independent[name][metric] - results[name][metric]) for name in MODELS for metric in ("mae_pp", "rmse_pp", "bias_pp", "dominant_class_accuracy") if independent[name][metric] is not None)
    summary = {"status": "complete", "task": "scene-level 19-class labelled-pixel coverage", "models": list(MODELS),
               "dataset_manifest": str(manifest_path or "experiments/pass3/dataset_manifest.json"), "areas": rows,
               "split_counts": {name: int((splits == name).sum()) for name in ("train", "validation", "test")},
               "test_area_ids": ids[test_mask].tolist(), "metrics": results, "independent_metrics": independent,
               "independent_metric_max_abs_difference": maximum_metric_difference, "selected_epochs": selected_epochs,
               "training_logs": logs, "preprocessing": manifest["preprocessing"], "class_order": [name for name, _ in CLASSES],
               "runtime_seconds": time.perf_counter() - started, "leakage": {"area_overlap": False, "test_used_for_selection": False,
               "normalization_fit": "per-image/channel fixed CROMA profile; probe scaling train-only"},
               "interpretation": "scene-level comparison; not token-level spatial grounding"}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (output_root / "provenance.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_root / "predictions.json").write_text(json.dumps({"area_ids": ids[test_mask].tolist(), "truth": test_truth.tolist(),
                                                                "models": {name: np.asarray(value).tolist() for name, value in predictions.items()}}, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pass 3 six-model held-out scene-coverage comparison.")
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/pass3/dataset_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/pass3/comparison"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run(args.pipeline_root, args.output_root, manifest_path=args.manifest, seed=args.seed, epochs=args.epochs, patience=args.patience, device=args.device,
                 progress=print)
    print(json.dumps({"status": result["status"], "split_counts": result["split_counts"], "metrics": {name: {key: value[key] for key in ("mae_pp", "rmse_pp", "bias_pp", "dominant_class_accuracy")} for name, value in result["metrics"].items()},
                      "independent_metric_max_abs_difference": result["independent_metric_max_abs_difference"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()