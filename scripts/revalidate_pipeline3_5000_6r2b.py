"""Audit-only full-population inference for Phase 6R.2B.

This harness deliberately reuses Pipeline 3's frozen cache loader, probe
loader, prediction function, state hash, and metric implementation.  It adds
only full-population prediction persistence and audit measurements; it does
not change preprocessing, features, labels, splits, model, or evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from run_pipeline3_5000_baseline import (
    DATASET_FINGERPRINT,
    SPLIT_FINGERPRINT,
    _load_probe,
    _metrics,
    _model_state_hash,
    load_cache,
)
from src.phase3_6_benchmark import predict_rows


EXPECTED_STATE_HASH = "85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634"
EXPECTED_FILE_HASH = "0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    checkpoint_file_hash = sha256(args.checkpoint)
    if checkpoint_file_hash != EXPECTED_FILE_HASH:
        raise ValueError(f"historical checkpoint file hash mismatch: {checkpoint_file_hash}")

    cache_started = time.perf_counter()
    data = load_cache(args.cache_root, include_test=True)
    cache_seconds = time.perf_counter() - cache_started
    ids = data["area_ids"]
    splits = data["splits"]
    truth = data["targets"].astype(np.float32)
    physical = data["physical"].astype(np.float32)
    joint = data["joint_croma"].astype(np.float32)
    hybrid = np.concatenate((physical, joint), axis=1)
    if hybrid.shape != (5000, 830) or physical.shape != (5000, 62) or joint.shape != (5000, 768):
        raise ValueError("frozen feature dimensions changed")
    if not all(np.isfinite(value).all() for value in (truth, physical, joint, hybrid)):
        raise ValueError("non-finite values in frozen feature cache")

    model = _load_probe(args.checkpoint, 830)
    state_hash = _model_state_hash(model)
    if state_hash != EXPECTED_STATE_HASH:
        raise ValueError(f"historical model state hash mismatch: {state_hash}")

    inference_started = time.perf_counter()
    predictions = predict_rows(model, torch.from_numpy(hybrid)).numpy()
    inference_seconds = time.perf_counter() - inference_started
    if predictions.shape != (5000, 19) or not np.isfinite(predictions).all():
        raise ValueError("historical checkpoint produced invalid predictions")

    test_mask = splits == "test"
    if int(test_mask.sum()) != 200:
        raise ValueError("sealed test split no longer contains 200 rows")
    metrics_started = time.perf_counter()
    test_metrics = _metrics(predictions[test_mask], truth[test_mask], ids[test_mask], 17)
    metrics_seconds = time.perf_counter() - metrics_started
    correct = int((predictions[test_mask].argmax(1) == truth[test_mask].argmax(1)).sum())

    historical_comparison = None
    if args.historical_predictions and args.historical_predictions.is_file():
        old = json.loads(args.historical_predictions.read_text(encoding="utf-8"))
        old_ids = np.asarray(old["area_ids"])
        old_values = np.asarray(old["models"]["hybrid"], dtype=np.float32)
        new_ids = ids[test_mask]
        if old_ids.tolist() != new_ids.tolist():
            raise ValueError("historical prediction IDs differ from the sealed test ordering")
        historical_comparison = {
            "path": str(args.historical_predictions),
            "sha256": sha256(args.historical_predictions),
            "maximum_absolute_prediction_difference": float(np.max(np.abs(old_values - predictions[test_mask]))),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_root / "historical_checkpoint_predictions_5000.npz"
    np.savez_compressed(
        prediction_path,
        area_ids=ids,
        splits=splits,
        predictions=predictions,
        truth=truth,
    )
    rss_after = process.memory_info().rss
    cache_manifest = args.cache_root / "manifest.json"
    summary = {
        "status": "complete",
        "execution_kind": "fresh probe inference over verified cached representations",
        "areas_attempted": 5000,
        "areas_completed": 5000,
        "areas_failed": 0,
        "areas_skipped": 0,
        "failed_area_ids": [],
        "failure_reasons": [],
        "split_counts": {name: int((splits == name).sum()) for name in ("train", "validation", "test")},
        "feature_shapes": {
            "physical": list(physical.shape),
            "joint_croma_gap": list(joint.shape),
            "hybrid": list(hybrid.shape),
            "predictions": list(predictions.shape),
        },
        "feature_order": "physical[0:62] then joint_croma_gap[62:830]",
        "output_classes": 19,
        "checkpoint": {
            "path": str(args.checkpoint),
            "file_sha256": checkpoint_file_hash,
            "state_dict_sha256": state_hash,
            "input_dimension": 830,
        },
        "cache": {
            "status": "verified_existing_cache",
            "manifest": str(cache_manifest),
            "manifest_sha256": sha256(cache_manifest),
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "all_shard_hashes_verified_by_loader": True,
            "historical_prediction_comparison": historical_comparison,
        },
        "sealed_test": {
            "count": 200,
            "correct": correct,
            "mae_pp": test_metrics["mae_pp"],
            "rmse_pp": test_metrics["rmse_pp"],
            "dominant_class_accuracy": test_metrics["dominant_class_accuracy"],
            "bootstrap_area_mae_95ci": test_metrics.get("bootstrap_area_mae_95ci"),
            "per_class": test_metrics.get("per_class"),
        },
        "timing_seconds": {
            "cache_validation_and_load": cache_seconds,
            "model_inference": inference_seconds,
            "metric_evaluation": metrics_seconds,
            "total": time.perf_counter() - started,
            "average_inference_per_area": inference_seconds / 5000,
            "inference_areas_per_second": 5000 / inference_seconds,
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "observed_rss_delta_bytes": rss_after - rss_before,
            "gpu_peak_bytes": 0,
        },
        "prediction_file": str(prediction_path),
        "prediction_file_sha256": sha256(prediction_path),
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "device": "cpu",
    }
    atomic_json(args.output_root / "historical_checkpoint_inference.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
