"""Build the machine-readable Phase 6R.2B reproducibility manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy as np
import pandas
import pyarrow
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase1_foundation import coverage_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--api-report", type=Path, required=True)
    args = parser.parse_args()

    inference = read(args.repo_output / "historical_checkpoint_inference.json")
    subset = read(args.repo_output / "cache_bypass_subset.json")
    training = read(args.training_root / "validation_results.json")
    test = read(args.training_root / "test_metrics.json")
    predictions_path = args.training_root / "test_predictions.json"
    predictions = read(predictions_path)
    api = read(args.api_report)

    ids = np.asarray(predictions["area_ids"])
    truth = np.asarray(predictions["truth"], dtype=np.float32)
    predicted = np.asarray(predictions["models"]["hybrid"], dtype=np.float32)
    recomputed = coverage_metrics(predicted, truth, ids, bootstrap_repeats=2000, seed=17)
    reported = test["metrics"]["hybrid"]
    metric_differences = {
        key: abs(float(recomputed[key]) - float(reported[key]))
        for key in ("mae_pp", "rmse_pp", "dominant_class_accuracy")
    }
    if max(metric_differences.values()) != 0:
        raise ValueError(f"saved prediction metrics do not reproduce exactly: {metric_differences}")

    fresh_checkpoint = Path(training["results"]["hybrid"]["checkpoint"])
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "rasterio": rasterio.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "cuda_tensor_operation": None,
    }
    leakage = {
        "data": {"status": "PASS", "evidence": "full audit: zero split intersections, cross-split raster duplicates, or cross-split spatial overlaps"},
        "label": {"status": "PASS", "evidence": "reference target is stored separately; 62D extractor accepts only optical/SAR and reference-only perturbation changed features by 0"},
        "feature": {"status": "PASS", "evidence": "physical/CROMA features use input rasters; standardization is fitted on train only"},
        "training": {"status": "PASS", "evidence": "fresh training loaded 4600 train + 200 validation and zero test rows; selection used validation loss"},
        "cache": {"status": "PASS", "evidence": "157 shard hashes verified; real-area bypass matched physical exactly and CROMA GAP within 1.13e-6"},
    }
    basis = {
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "environment": environment,
        "dataset_fingerprint": inference["dataset_fingerprint"],
        "split_fingerprint": inference["split_fingerprint"],
        "historical_checkpoint_file_sha256": sha256(args.historical_checkpoint),
        "historical_checkpoint_state_sha256": inference["checkpoint"]["state_dict_sha256"],
        "fresh_checkpoint_file_sha256": sha256(fresh_checkpoint),
        "fresh_checkpoint_state_sha256": training["results"]["hybrid"]["state_dict_sha256"],
        "configuration_sha256": sha256(args.baseline_config),
        "seed": 17,
        "selected_epoch": training["results"]["hybrid"]["selected_epoch"],
        "historical_checkpoint_test_metrics": {
            key: inference["sealed_test"][key]
            for key in ("mae_pp", "rmse_pp", "dominant_class_accuracy", "correct", "count")
        },
        "fresh_training_test_metrics": {
            "mae_pp": reported["mae_pp"],
            "rmse_pp": reported["rmse_pp"],
            "dominant_class_accuracy": reported["dominant_class_accuracy"],
            "correct": int((predicted.argmax(1) == truth.argmax(1)).sum()),
            "count": len(ids),
        },
        "cache_manifest_sha256": inference["cache"]["manifest_sha256"],
        "cache_bypass_maximum_croma_gap_difference": max(subset["croma"]["fresh_vs_cache_maximum_absolute_differences"].values()),
        "croma_repeat_maximum_difference": max(subset["croma"]["repeat_maximum_absolute_differences"].values()),
        "saved_prediction_metric_maximum_difference": max(metric_differences.values()),
        "api_http_equivalent_status": 422,
        "api_result_status": api["status"],
        "api_registry_connected": False,
        "leakage": leakage,
    }
    manifest = {
        "status": "complete_with_critical_connectivity_failure",
        "final_verdict": "RED",
        "scientific_revalidation_fingerprint": canonical_hash(basis),
        "fingerprint_algorithm": "sha256(canonical_json(scientific_basis))",
        "basis": basis,
        "environment": environment,
        "timing_seconds": {
            "dataset_audit": read(Path("experiments/pipeline3_5000/audit_summary.json"))["timing_seconds"],
            "historical_checkpoint_inference": inference["timing_seconds"],
            "cache_bypass_subset": subset["timing_seconds"],
            "fresh_training": training["runtime_seconds"],
            "fresh_test_scoring_persisted_run": test["runtime_seconds"],
            "api_rejected_request": api["runtime_seconds"],
        },
        "memory": {
            "historical_checkpoint_inference": inference["memory"],
            "cache_bypass_subset": subset["memory"],
            "fresh_training_peak_gpu_bytes": training["peak_gpu_memory_bytes"],
        },
        "saved_prediction_recomputation": {
            "prediction_file": str(predictions_path),
            "prediction_file_sha256": sha256(predictions_path),
            "metric_differences": metric_differences,
            "exact_match": True,
        },
        "artifacts": {
            "historical_full_predictions_sha256": inference["prediction_file_sha256"],
            "validation_results_sha256": sha256(args.training_root / "validation_results.json"),
            "fresh_checkpoint_sha256": sha256(fresh_checkpoint),
            "test_metrics_sha256": sha256(args.training_root / "test_metrics.json"),
            "test_receipt_sha256": sha256(args.training_root / "test_receipt.json"),
            "test_scientific_fingerprint": read(args.training_root / "scientific_fingerprint.json")["fingerprint"],
            "api_report": str(args.api_report),
            "api_report_sha256": sha256(args.api_report),
        },
        "leakage_audit": leakage,
        "connectivity": {
            "status": "FAIL",
            "api_uses_capability_registry": False,
            "api_accepts_full_identity_5000_dataset_root": False,
            "trained_830d_probe_connected_to_api_evidence_and_task_result": False,
            "observed_api_result": "HTTP 422 rejected; local discovery reported duplicate bands/unavailable imagery",
            "direct_small_sample_registry_path": "PASS in two real-data tests, but it is not the 5,000-area trained probe path",
        },
        "provenance_limitation": "The restored 4,000 S2 areas have acquisition receipts; the original 1,000 do not.",
        "sealed_test_operational_note": "A first identical scoring attempt computed in memory but failed before any output write due sandbox permission; no result was observed or used. The frozen command was rerun only to persist artifacts.",
    }
    atomic_json(args.repo_output / "reproducibility.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "final_verdict": manifest["final_verdict"],
        "scientific_revalidation_fingerprint": manifest["scientific_revalidation_fingerprint"],
        "saved_prediction_metric_maximum_difference": max(metric_differences.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
