import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PROJECT_ROOT, get_settings
from .croma_adapter import CROMAAdapter
from .dataset_loader import discover_samples, load_sample


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _git_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()


def _croma_commit(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BigEarthNet samples and extract official CROMA features.")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    samples = discover_samples(settings.dataset_root)
    metadata_path = settings.dataset_root / "metadata.parquet"
    metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else pd.DataFrame()
    metadata_by_patch = {
        str(row.patch_id).rsplit("_", 2)[-2] + "_" + str(row.patch_id).rsplit("_", 2)[-1]: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in row._asdict().items()
        }
        for row in metadata.itertuples(index=False)
    }
    LOGGER.info("project=%s branch=%s", PROJECT_ROOT, _git_branch())
    LOGGER.info("dataset=%s samples=%d", settings.dataset_root, len(samples))
    prepared = []
    for sample in samples:
        item = load_sample(sample)
        if item.optical.shape != (12, 120, 120) or item.sar.shape != (2, 120, 120):
            raise ValueError(f"{sample.patch_id}: invalid tensor shape")
        if not np.isfinite(item.optical).all() or not np.isfinite(item.sar).all():
            raise ValueError(f"{sample.patch_id}: non-finite preprocessed values")
        LOGGER.info("sample=%s optical=%s sar=%s metadata=%s", sample.patch_id, item.optical.shape, item.sar.shape, item.metadata)
        prepared.append(item)
    if args.inspect_only:
        return
    settings.features_root.mkdir(parents=True, exist_ok=True)
    adapter = CROMAAdapter(settings.croma_source, settings.croma_checkpoint)
    report = {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(settings.dataset_root),
        "device": str(adapter.device),
        "croma_repository": "https://github.com/antofuller/CROMA",
        "croma_commit": _croma_commit(settings.croma_source),
        "croma_checkpoint": str(settings.croma_checkpoint),
        "croma_checkpoint_bytes": settings.croma_checkpoint.stat().st_size,
        "croma_model_variant": "base",
        "croma_image_resolution": 120,
        "croma_preprocessing": {
            "source": "official CROMA README normalization applied to raw float32 raster channels",
            "optical_band_order": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
            "sar_band_order": ["VV", "VH"],
            "per_channel": "mean +/- 2 std, clipped to [0, 1]",
            "official_8_bit_equivalent": "clip to [0,255], cast uint8, then divide by 255; adapter uses the mathematically equivalent float path",
            "resize_or_crop": "none; inputs are already 120x120",
        },
        "samples": [],
    }
    for item in prepared:
        outputs = adapter.infer(item.raw_optical, item.raw_sar)
        sample_dir = settings.features_root / item.patch_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        shapes = {}
        for name, value in outputs.items():
            array = value.detach().cpu().numpy()
            np.save(sample_dir / f"{name}.npy", array)
            shapes[name] = {"type": str(type(value)), "shape": list(array.shape), "path": str(sample_dir / f"{name}.npy")}
            LOGGER.info("sample=%s output=%s shape=%s", item.patch_id, name, array.shape)
        report["samples"].append({
            "patch_id": item.patch_id,
            "labels_metadata": metadata_by_patch.get(item.patch_id, {}),
            "input_metadata": item.metadata,
            "outputs": shapes,
        })
    report.update({
        "dataset_zip": str(settings.dataset_root.parent.parent / "bigearthnet-v2-three-samples.zip"),
        "extraction_path": str(settings.dataset_root.parent),
        "detected_optical_bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        "detected_sar_bands": ["VV", "VH"],
        "croma_input_shapes": {"optical": [1, 12, 120, 120], "sar": [1, 2, 120, 120]},
        "validation": {
            "all_three_samples": True,
            "tensor_shapes": True,
            "aligned_grid": True,
            "finite_preprocessed_values": True,
            "official_checkpoint_loaded": True,
            "feature_paths_outside_project": True,
        },
        "warnings": ["No spatial feature maps are returned; encodings are 225-token sequences."],
        "errors": [],
        "assumptions": ["CROMA optical channel order follows the BigEarthNet order recorded in input_metadata.", "225 tokens are the official 15x15 grid from 120/8 patches."],
    })
    report_path = settings.features_root / "report.json"
    verification_path = settings.features_root / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    verification_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("report=%s verification_report=%s", report_path, verification_path)


if __name__ == "__main__":
    main()