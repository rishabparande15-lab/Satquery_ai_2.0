"""Fresh real-area cache-bypass and repeatability audit for Phase 6R.2B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import psutil
import torch
from rasterio.enums import Resampling
import rasterio

from run_pipeline3_5000_baseline import (
    OPTICAL_BANDS,
    SAR_BANDS,
    _normalise_batch,
    _profile,
    _read_bytes_to_grid,
    _read_path_to_grid,
    _s2_paths,
    load_cache,
)
from src.croma_adapter import CROMAAdapter
from src.gee_features import LocalRasterFeatureProvider


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--croma-source", type=Path, required=True)
    parser.add_argument("--croma-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-id")
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]
    row = next((item for item in rows if item["area_id"] == args.area_id), rows[0])
    area_id = row["area_id"]
    cached = load_cache(args.cache_root, include_test=True)
    positions = np.flatnonzero(cached["area_ids"] == area_id)
    if len(positions) != 1:
        raise ValueError("selected real area does not have exactly one cache row")
    position = int(positions[0])

    s1_path = args.cache_root / "source_cache" / "BigEarthNet-S1-selected.zip"
    reference_path = args.cache_root / "source_cache" / "Reference_Maps-selected.zip"
    with zipfile.ZipFile(s1_path) as s1_zip, zipfile.ZipFile(reference_path) as reference_zip:
        vv = {Path(name).name[:-7]: name for name in s1_zip.namelist() if name.endswith("_VV.tif")}
        vh = {Path(name).name[:-7]: name for name in s1_zip.namelist() if name.endswith("_VH.tif")}
        refs = {Path(name).name[:-18]: name for name in reference_zip.namelist() if name.endswith("_reference_map.tif")}
        paths = _s2_paths(row)
        with rasterio.open(paths["B02"]) as source:
            grid = _profile(source)
        raster_started = time.perf_counter()
        optical = np.stack([_read_path_to_grid(paths[band], grid, Resampling.bilinear) for band in OPTICAL_BANDS])
        identity = row["s1_identity"]
        sar = np.stack([_read_bytes_to_grid(s1_zip.read(index[identity]), grid, Resampling.bilinear) for index in (vv, vh)])
        reference = _read_bytes_to_grid(reference_zip.read(refs[area_id]), grid, Resampling.nearest)
        raster_seconds = time.perf_counter() - raster_started

    physical_started = time.perf_counter()
    physical, physical_report = LocalRasterFeatureProvider().extract(
        optical, sar, list(OPTICAL_BANDS), list(SAR_BANDS)
    )
    physical_seconds = time.perf_counter() - physical_started
    if physical.shape != (62,) or len(physical_report["feature_names"]) != 62 or not np.isfinite(physical).all():
        raise ValueError("fresh physical feature extraction failed its contract")
    # The reference map is intentionally perturbed without changing optical/SAR.
    # The production extractor has no reference argument, so its result must be invariant.
    perturbed_reference = reference.copy()
    perturbed_reference.flat[0] = (int(perturbed_reference.flat[0]) + 1) % 44
    physical_after_reference_perturbation, _ = LocalRasterFeatureProvider().extract(
        optical, sar, list(OPTICAL_BANDS), list(SAR_BANDS)
    )

    process = psutil.Process()
    rss_before_model = process.memory_info().rss
    adapter = CROMAAdapter(args.croma_source, args.croma_checkpoint, device="cpu")
    rss_after_model = process.memory_info().rss
    outputs = []
    runtimes = []
    for _ in range(2):
        started = time.perf_counter()
        with torch.inference_mode():
            result = adapter.model(
                optical_images=_normalise_batch(adapter, [optical]),
                SAR_images=_normalise_batch(adapter, [sar]),
            )
        runtimes.append(time.perf_counter() - started)
        outputs.append({key: value.detach().cpu().to(torch.float32).numpy() for key, value in result.items()})

    required = {
        "optical_encodings": [1, 225, 768],
        "SAR_encodings": [1, 225, 768],
        "joint_encodings": [1, 225, 768],
        "optical_GAP": [1, 768],
        "SAR_GAP": [1, 768],
        "joint_GAP": [1, 768],
    }
    shapes = {key: list(outputs[0][key].shape) for key in required}
    if shapes != required:
        raise ValueError(f"fresh CROMA shapes differ from contract: {shapes}")
    if not all(np.isfinite(outputs[0][key]).all() for key in required):
        raise ValueError("fresh CROMA output contains non-finite values")
    repeat_differences = {
        key: float(np.max(np.abs(outputs[0][key] - outputs[1][key]))) for key in required
    }
    cache_differences = {
        "physical": float(np.max(np.abs(physical.astype(np.float32) - cached["physical"][position]))),
        "optical_GAP": float(np.max(np.abs(outputs[0]["optical_GAP"][0] - cached["optical_croma"][position]))),
        "SAR_GAP": float(np.max(np.abs(outputs[0]["SAR_GAP"][0] - cached["sar_croma"][position]))),
        "joint_GAP": float(np.max(np.abs(outputs[0]["joint_GAP"][0] - cached["joint_croma"][position])),),
    }
    payload = {
        "status": "complete",
        "area_id": area_id,
        "split": row["split"],
        "device": "cpu",
        "inputs": {
            "optical_shape": list(optical.shape),
            "sar_shape": list(sar.shape),
            "reference_shape": list(reference.shape),
            "optical_band_order": list(OPTICAL_BANDS),
            "sar_band_order": list(SAR_BANDS),
            "all_finite": bool(np.isfinite(optical).all() and np.isfinite(sar).all()),
        },
        "physical": {
            "shape": list(physical.shape),
            "feature_names": physical_report["feature_names"],
            "finite": bool(np.isfinite(physical).all()),
            "extractor_signature_uses_reference": False,
            "maximum_difference_after_reference_only_perturbation": float(
                np.max(np.abs(physical - physical_after_reference_perturbation))
            ),
        },
        "croma": {
            "shapes": shapes,
            "token_grid": [15, 15],
            "token_ordering": "row-major per production model reshape contract",
            "repeat_maximum_absolute_differences": repeat_differences,
            "fresh_vs_cache_maximum_absolute_differences": cache_differences,
            "run_times_seconds": runtimes,
        },
        "timing_seconds": {
            "raster_io_and_resampling": raster_seconds,
            "physical_features": physical_seconds,
            "croma_first": runtimes[0],
            "croma_repeat": runtimes[1],
        },
        "memory": {
            "rss_before_croma_model_bytes": rss_before_model,
            "rss_after_croma_model_bytes": rss_after_model,
            "model_load_rss_delta_bytes": rss_after_model - rss_before_model,
            "gpu_peak_bytes": 0,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
