from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .croma_adapter import CROMAAdapter
from .config import get_settings
from .dataset_loader import discover_samples, load_sample
from .gee_features import LocalRasterFeatureProvider
from .hybrid_fusion import HybridFusion
from .phase1_foundation import SampleIdentity, prepare_batch, sha256
from .phase2_integration import adapt_phase1_output


def _metadata_by_patch(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    result = {}
    for row in frame.itertuples(index=False):
        patch = str(getattr(row, "patch_id"))
        match = re.search(r"(\d+_\d+)$", patch)
        if match:
            result[match.group(1)] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row._asdict().items()}
    return result


def run(dataset_root: Path, output_root: Path, device: str | None = None, split: str = "test") -> dict:
    started = time.perf_counter()
    samples = discover_samples(dataset_root)
    metadata = _metadata_by_patch(dataset_root / "metadata.parquet")
    settings = get_settings()
    adapter = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device=device)
    physical_provider = LocalRasterFeatureProvider()
    torch.manual_seed(42)
    fusion = None
    reports = []
    for sample in samples:
        item = load_sample(sample)
        identity = SampleIdentity(
            sample.patch_id,
            sample.sar_paths["VV"].stem,
            sample.optical_paths["B02"].stem,
            split,
            {"source": "BigEarthNet fixture", "grid": item.metadata},
            item.metadata["crs"],
            tuple(item.metadata["bounds"]),
            tuple(item.metadata["resolution"]),
            {"dataset_root": str(dataset_root)},
        )
        phase1_batch = prepare_batch(torch.from_numpy(item.raw_optical[None]), torch.from_numpy(item.raw_sar[None]), [identity])
        gee_vector, gee_report = physical_provider.extract(item.raw_optical, item.raw_sar, item.metadata["optical_band_order"], item.metadata["sar_band_order"])
        gee_report = {**gee_report, "sample_id": sample.patch_id}
        croma_outputs = adapter.infer(item.raw_optical, item.raw_sar)
        phase2_input = adapt_phase1_output(
            phase1_batch,
            croma_outputs,
            gee_vector[None, :],
            [gee_report],
            croma_provenance={"checkpoint": str(settings.croma_checkpoint), "checkpoint_sha256": sha256(settings.croma_checkpoint)},
            experiment_configuration={"pipeline": "hybrid_pipeline", "device": str(adapter.device)},
        )
        croma_vector = phase2_input.pooled_croma_features[0]
        if fusion is None:
            fusion = HybridFusion(gee_vector.size, croma_vector.size).to(adapter.device).eval()
        elif fusion.gee_projection[0].normalized_shape[0] != gee_vector.size or fusion.croma_projection[0].normalized_shape[0] != croma_vector.size:
            raise ValueError(f"{sample.patch_id}: feature dimensions changed across samples")
        with torch.no_grad():
            hybrid = fusion(torch.from_numpy(phase2_input.physical_features[0]).to(adapter.device).unsqueeze(0), torch.from_numpy(croma_vector).to(adapter.device).unsqueeze(0)).cpu().numpy()[0]
        sample_root = output_root / sample.patch_id
        croma_root = sample_root / "croma_features"
        hybrid_root = sample_root / "hybrid_features"
        croma_root.mkdir(parents=True, exist_ok=True); hybrid_root.mkdir(parents=True, exist_ok=True)
        croma_paths = {}
        for name in ("optical_GAP", "SAR_GAP", "joint_GAP"):
            path = croma_root / f"{name}.npy"; np.save(path, croma_outputs[name].detach().cpu().numpy()); croma_paths[name] = str(path)
        hybrid_path = hybrid_root / "hybrid_vector.npy"; np.save(hybrid_path, hybrid)
        report = {
            "sample_id": sample.patch_id,
            "source": "local BigEarthNet proof-of-concept dataset",
            "input": {"optical_shape": list(item.raw_optical.shape), "sar_shape": list(item.raw_sar.shape), "metadata": item.metadata, "labels": metadata.get(sample.patch_id, {}), "identity": phase2_input.provenance[0]["identity"]},
            "gee_features": {"dimension": gee_report["dimension"], "feature_names": gee_report["feature_names"], "feature_values": gee_report["feature_values"], "valid_pixel_counts": gee_report["valid_pixel_counts"], "formulas": gee_report["formulas"]},
            "croma": {"input_shapes": {"optical": [1, *item.raw_optical.shape], "sar": [1, *item.raw_sar.shape]}, "output_shapes": {name: list(value.shape) for name, value in croma_outputs.items()}, "pooled_feature_dimension": int(croma_vector.size), "pooled_paths": croma_paths, "token_order": "row-major 15x15", "provenance": phase2_input.provenance[0]["croma"]},
            "fusion": {"gee_dimension": int(gee_vector.size), "croma_dimension": int(croma_vector.size), "hybrid_dimension": int(hybrid.size), "weights": "deterministic seeded untrained prototype fusion; no accuracy claim" , "path": str(hybrid_path)},
            "validation": {"status": "passed", "finite_physical_features": bool(np.isfinite(gee_vector).all()), "finite_croma_features": bool(np.isfinite(croma_vector).all()), "finite_hybrid_features": bool(np.isfinite(hybrid).all()), "metadata_present": bool(metadata.get(sample.patch_id)), "crs": item.metadata["crs"], "resolution": item.metadata["resolution"], "phase2_adapter": True},
            "temporal_pairing": {"valid": False, "reason": "The three provided samples have no before/after temporal pairing metadata."},
            "analysis": {"status": "feature extraction and fusion validated; downstream classifier training deferred until more labeled data", "evidence": ["optical spectral statistics and indices", "SAR VV/VH statistics and relationship", "CROMA optical, SAR, and joint pooled representations"]},
        }
        sample_root.mkdir(parents=True, exist_ok=True)
        (sample_root / "gee_features.json").write_text(json.dumps(report["gee_features"], indent=2), encoding="utf-8")
        (sample_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
    final = {"dataset_root": str(dataset_root), "dataset_zip": str(dataset_root.parent.parent / "bigearthnet-v2-three-samples.zip"), "samples_processed": len(reports), "sample_reports": reports, "device": str(adapter.device), "official_croma_checkpoint": str(settings.croma_checkpoint), "runtime_seconds": time.perf_counter() - started, "feature_schema": {"gee_physical": {"dimension": 62, "normalization": "raw raster statistics; no learned normalization", "missing_value_behavior": "fail on missing required bands or no finite pixels"}, "croma_pooled": {"dimension": 2304, "normalization": "official CROMA per-channel mean +/- 2 std clipped to [0,1]", "missing_value_behavior": "fail on missing representation"}, "hybrid": {"dimension": fusion.output_dim if fusion else None, "normalization": "LayerNorm in fusion projections", "missing_value_behavior": "fail on non-finite features"}}, "temporal_pairing": {"valid": False, "reason": "No temporal pair is represented by the three sample IDs."}, "training": {"performed": False, "reason": "Three samples are insufficient for meaningful supervised accuracy claims."}, "errors": []}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local BigEarthNet -> GEE features -> CROMA -> hybrid proof of concept.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/outputs/hybrid_poc"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test", help="Explicit split for datasets without a split manifest")
    args = parser.parse_args()
    settings = get_settings()
    result = run(args.dataset_root or settings.dataset_root, args.output_root, args.device, args.split)
    print(json.dumps({"samples_processed": result["samples_processed"], "device": result["device"], "runtime_seconds": result["runtime_seconds"], "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
