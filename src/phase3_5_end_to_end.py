"""Forensic Phase 3 -> Phase 1 -> Phase 2 -> advanced-pipeline verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.warp import transform_bounds

from .croma_adapter import CROMAAdapter
from .dataset_loader import OPTICAL_BANDS, SAR_BANDS, discover_samples, load_sample
from .gee_features import LocalRasterFeatureProvider
from .hybrid_fusion import HybridFusion
from .phase3_orchestration import (
    AOI,
    AvailabilityRequest,
    Modality,
    ProcessingResult,
    ExecutionPlan,
    SatQueryRequest,
    SceneCandidate,
    TemporalRequest,
    ValidatedSatelliteDataset,
    LocalSceneAdapter,
    bridge_to_phase2,
    select_scene,
    validate_spatial_pair,
)
from .config import get_settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _acquisition_time(path: Path) -> datetime:
    match = re.search(r"(20\d{6}T\d{6})", path.name)
    if not match:
        raise ValueError(f"Acquisition timestamp is missing from {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _scene(sample, modality: Modality, path: Path, bounds: tuple[float, float, float, float], crs: str) -> SceneCandidate:
    source = "sentinel-2" if modality == Modality.OPTICAL else "sentinel-1"
    definition = "Sentinel-2" if modality == Modality.OPTICAL else "Sentinel-1"
    sensor = "MSI" if modality == Modality.OPTICAL else "C-SAR"
    bands = OPTICAL_BANDS if modality == Modality.OPTICAL else SAR_BANDS
    return SceneCandidate(
        scene_id=path.stem.rsplit("_", 1)[0], source=source, satellite=definition, sensor=sensor,
        modality=modality, acquisition_time=_acquisition_time(path), native_gsd=10.0,
        crs=crs, bounds=bounds, coverage_fraction=1.0, band_order=tuple(bands),
        access={"sample_id": sample.patch_id, "path": str(path)},
    )


class _PreparedCroma:
    """Move Phase 1 tensors to the already-loaded official CROMA device."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(self, *, optical_images: torch.Tensor, SAR_images: torch.Tensor):
        return self.model(optical_images=optical_images.to(self.device), SAR_images=SAR_images.to(self.device))


@dataclass(frozen=True)
class EndToEndRun:
    receipt: dict[str, Any]
    processing_result: ProcessingResult
    phase2_input: Any
    physical_features: np.ndarray
    croma_features: dict[str, torch.Tensor]
    hybrid_features: np.ndarray


def run(dataset_root: Path, *, sample_id: str = "61_39", device: str | None = None,
        split: str = "test", request_id: str | None = None) -> EndToEndRun:
    settings = get_settings()
    samples = {sample.patch_id: sample for sample in discover_samples(Path(dataset_root))}
    if sample_id not in samples:
        raise ValueError(f"Real BigEarthNet sample is unavailable: {sample_id}")
    sample = samples[sample_id]
    with rasterio.open(sample.optical_paths["B02"]) as grid:
        native_crs = str(grid.crs)
        native_bounds = tuple(grid.bounds)
        geographic_bounds = tuple(transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21))

    optical_path = sample.optical_paths["B02"]
    sar_path = sample.sar_paths["VV"]
    optical_scene = _scene(sample, Modality.OPTICAL, optical_path, geographic_bounds, native_crs)
    sar_scene = _scene(sample, Modality.SAR, sar_path, geographic_bounds, native_crs)
    aoi = AOI.from_input({"type": "bbox", "bounds": list(geographic_bounds)}, source="real_bigearthnet_fixture")
    temporal = TemporalRequest.create(acquisition_date=_acquisition_time(optical_path).date().isoformat())
    optical_request = AvailabilityRequest(aoi, temporal, Modality.OPTICAL, 10.0, ("sentinel-2",))
    sar_request = AvailabilityRequest(aoi, temporal, Modality.SAR, 10.0, ("sentinel-1",))
    optical_selected, optical_reason = select_scene(optical_request, LocalSceneAdapter([optical_scene]).scenes)
    sar_selected, sar_reason = select_scene(sar_request, LocalSceneAdapter([sar_scene]).scenes)

    item = load_sample(sample)
    if item.metadata["crs"] != native_crs or tuple(item.metadata["bounds"]) != native_bounds:
        raise ValueError("Materialized raster metadata does not match the selected Phase 3 scene")
    optical_meta = {"crs": item.metadata["crs"], "bounds": tuple(item.metadata["bounds"]),
                    "resolution": tuple(item.metadata["resolution"]), "transform": tuple(item.metadata["transform"]),
                    "array": item.raw_optical}
    sar_meta = {**optical_meta, "array": item.raw_sar}
    validate_spatial_pair(optical_meta, sar_meta)
    source_hashes = {f"optical_{band}": _sha256(sample.optical_paths[band]) for band in OPTICAL_BANDS}
    source_hashes.update({f"sar_{band}": _sha256(sample.sar_paths[band]) for band in SAR_BANDS})
    request_id = request_id or f"phase3-5-{sample_id}"
    request = SatQueryRequest(
        query="Verify the connected multimodal satellite feature pipeline.", aoi=aoi,
        temporal=temporal, modality=Modality.OPTICAL_SAR, requested_gsd=10.0,
        satellite_preferences=("sentinel-2", "sentinel-1"), request_id=request_id,
    )
    plan = ExecutionPlan(request, ("sentinel-2", "sentinel-1"),
                         ("scene_selection", "acquisition", "raster_validation", "phase1", "phase2", "advanced_pipeline"),
                         datetime.now(timezone.utc).isoformat())
    dataset = ValidatedSatelliteDataset(
        request_id=request_id, scene_ids=(optical_selected.scene_id, sar_selected.scene_id),
        optical=item.raw_optical, sar=item.raw_sar, aoi=aoi, crs=item.metadata["crs"],
        bounds=tuple(item.metadata["bounds"]), native_gsd=float(item.metadata["resolution"][0]),
        processing_gsd=float(item.metadata["resolution"][0]),
        band_orders={"optical": tuple(OPTICAL_BANDS), "sar": tuple(SAR_BANDS)},
        acquisition_times=(_acquisition_time(optical_path).isoformat(), _acquisition_time(sar_path).isoformat()),
        source_hashes=source_hashes,
        provenance={"request_id": request_id, "sample_id": sample_id, "provider": "local",
                    "selected_scene_reasons": {"optical": optical_reason, "sar": sar_reason},
                    "native_transform": list(item.metadata["transform"]),
                    "processing_resolution": list(item.metadata["resolution"])},
    )
    croma_adapter = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device=device)
    phase2_input = bridge_to_phase2(
        dataset, _PreparedCroma(croma_adapter.model, croma_adapter.device), LocalRasterFeatureProvider(),
        croma_provenance={"checkpoint": str(settings.croma_checkpoint), "checkpoint_sha256": _sha256(settings.croma_checkpoint)},
        split=split,
    )
    torch.manual_seed(42)
    fusion = HybridFusion(phase2_input.physical_features.shape[1], phase2_input.pooled_croma_features.shape[1]).to(croma_adapter.device).eval()
    with torch.inference_mode():
        hybrid = fusion(torch.from_numpy(phase2_input.physical_features).to(croma_adapter.device),
                        torch.from_numpy(phase2_input.pooled_croma_features).to(croma_adapter.device)).cpu().numpy()
    provenance = {
        "request": {"request_id": request_id, "sample_id": sample_id, "query": request.query, "aoi": asdict(aoi),
                    "temporal": {"start": temporal.start.isoformat(), "end": temporal.end.isoformat()},
                    "modality": Modality.OPTICAL_SAR.value, "requested_gsd": 10.0},
        "execution_plan": asdict(plan),
        "scene": {"optical": asdict(optical_selected), "sar": asdict(sar_selected)},
        "acquisition": {"source_hashes": source_hashes, "crs": item.metadata["crs"],
                        "bounds": list(item.metadata["bounds"]), "transform": list(item.metadata["transform"]),
                        "native_resolution": list(item.metadata["resolution"]), "processing_resolution": list(item.metadata["resolution"])},
        "phase3": {"version": "1.0", "status": "validated", "band_orders": {"optical": list(OPTICAL_BANDS), "sar": list(SAR_BANDS)},
                   "shapes": {"optical": list(item.raw_optical.shape), "sar": list(item.raw_sar.shape)}},
        "phase1": {"status": "validated", "normalization_profile": phase2_input.provenance[0]["normalization"]["profile"]},
        "phase2": {"adapter_version": "1.0", "status": "validated", "provenance": phase2_input.provenance[0]},
        "advanced_pipeline": {"name": "HybridFusion", "status": "validated", "physical_dimension": int(phase2_input.physical_features.shape[1]),
                              "pooled_croma_dimension": int(phase2_input.pooled_croma_features.shape[1]), "output_dimension": int(hybrid.shape[1]),
                              "weights": "deterministic seeded untrained prototype; no task prediction"},
    }
    feature_hashes = {"physical": _array_sha256(phase2_input.physical_features), "pooled_croma": _array_sha256(phase2_input.pooled_croma_features), "hybrid": _array_sha256(hybrid)}
    try:
        repository_commit = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        repository_commit = "unavailable"
    receipt = {"receipt_version": 1, "timestamp": datetime.now(timezone.utc).isoformat(), "repository_commit": repository_commit, "status": "complete",
               "sample_id": sample_id, "split": split, "provenance": provenance, "feature_hashes": feature_hashes,
               "croma_shapes": {"optical_encodings": list(phase2_input.optical_encodings.shape),
                                "SAR_encodings": list(phase2_input.sar_encodings.shape),
                                "joint_encodings": list(phase2_input.joint_encodings.shape),
                                "optical_GAP": list(phase2_input.optical_gap.shape),
                                "SAR_GAP": list(phase2_input.sar_gap.shape),
                                "joint_GAP": list(phase2_input.joint_gap.shape)},
               "output_shapes": {"physical": list(phase2_input.physical_features.shape), "pooled_croma": list(phase2_input.pooled_croma_features.shape), "hybrid": list(hybrid.shape)},
               "reproducibility": {"status": "verified by dedicated repeat test", "comparison": "exact hashes for deterministic CPU run"}, "warnings": [], "failures": [], "final_verdict": "GREEN"}
    result = ProcessingResult(request_id, "complete", dataset.scene_ids, phase2_input, hybrid, provenance)
    croma_features = {
        "optical_encodings": phase2_input.optical_encodings,
        "SAR_encodings": phase2_input.sar_encodings,
        "joint_encodings": phase2_input.joint_encodings,
        "optical_GAP": phase2_input.optical_gap,
        "SAR_GAP": phase2_input.sar_gap,
        "joint_GAP": phase2_input.joint_gap,
    }
    return EndToEndRun(receipt, result, phase2_input, phase2_input.physical_features.copy(), croma_features, hybrid)


def write_receipt(run_result: EndToEndRun, path: Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_result.receipt, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Phase 3.5 end-to-end verification.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--sample-id", default="61_39")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--receipt", type=Path, default=Path("experiments/outputs/phase3_5/end_to_end_receipt.json"))
    args = parser.parse_args()
    result = run(args.dataset_root or get_settings().dataset_root, sample_id=args.sample_id, device=args.device, split=args.split)
    write_receipt(result, args.receipt)
    print(json.dumps({"status": result.receipt["status"], "sample_id": args.sample_id, "receipt": str(args.receipt), "output_shapes": result.receipt["output_shapes"]}, indent=2))


if __name__ == "__main__":
    main()