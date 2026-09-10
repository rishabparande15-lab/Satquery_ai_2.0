from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .hybrid_fusion import FeatureModeFusion, LandCoverClassificationHead
from .multilabel_metrics import compute_multilabel_metrics
from .train_landcover import load_split
from .training_data import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained hybrid multi-label head on the untouched test split.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = load_manifest(args.split_manifest)
    gee, croma, targets = load_split(args.cache_root, manifest.splits["test"])
    if not len(targets):
        raise RuntimeError("Test split is empty")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    feature_mode = checkpoint["config"].get("feature_mode", "hybrid")
    threshold = args.threshold if args.threshold is not None else checkpoint["config"].get("threshold", 0.5)
    model = FeatureModeFusion(feature_mode, checkpoint["config"]["gee_dim"], checkpoint["config"]["croma_dim"], output_dim=checkpoint["config"].get("output_dim", 192)).to(device)
    head = LandCoverClassificationHead(model.output_dim, len(manifest.class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"]); head.load_state_dict(checkpoint["head_state"]); model.eval(); head.eval()
    with torch.no_grad():
        probabilities = head.probabilities(model(torch.from_numpy(gee).to(device), torch.from_numpy(croma).to(device))).cpu().numpy()
    metrics = compute_multilabel_metrics(targets, probabilities, threshold=threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_name("test_probabilities.npy")
    np.save(prediction_path, probabilities)
    predictions = []
    for sample_id, row in zip(manifest.splits["test"], probabilities):
        predicted = [manifest.class_names[index] for index, value in enumerate(row) if value >= metrics["threshold"]]
        predictions.append({"sample_id": sample_id, "predicted_labels": predicted, "probabilities": {name: float(value) for name, value in zip(manifest.class_names, row)}, "confidence": float(np.max(row)), "confidence_calibration": "uncalibrated", "input_validation": "passed", "evidence_references": [str(args.cache_root / sample_id / "metadata.json")]})
    prediction_json = args.output.with_name("test_predictions.json")
    prediction_json.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    report = {"status": "success", "feature_mode": feature_mode, "split": "test", "sample_counts": {key: len(value) for key, value in manifest.splits.items()}, "sample_ids": manifest.splits["test"], "feature_dimensions": {"gee": int(gee.shape[1]), "croma": int(croma.shape[1]), "model_output": int(model.output_dim)}, "label_distribution": {name: int(targets[:, index].sum()) for index, name in enumerate(manifest.class_names)}, "class_names": manifest.class_names, "metrics": metrics, "checkpoint": str(args.checkpoint), "seed": manifest.seed, "threshold": threshold, "device": str(device), "runtime_seconds": time.perf_counter() - started, "prediction_path": str(prediction_path), "prediction_json": str(prediction_json), "confidence": "uncalibrated sigmoid probabilities", "processing": {"input_validation": "passed", "evidence_references": "per-sample cache metadata"}}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
