"""Strict BigEarthNet identity resolution and the validated Pipeline 3 scene probe.

This module is intentionally inference-only.  It reuses the frozen scientific
preprocessing/model contracts and never converts the scene prediction into
token predictions or semantic claims.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4
import zipfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject
import torch

from .config import PROJECT_ROOT, get_settings
from .croma_adapter import CROMAAdapter
from .dataset_loader import OPTICAL_BANDS, SAR_BANDS
from .evidence_schema import build_spatial_evidence
from .gee_features import LocalRasterFeatureProvider
from .interpretation_adapter import interpret_evidence, technical_companion, validate_interpretation
from .phase1_foundation import CLASSES
from .phase3_6_benchmark import ProbeHead


MANIFEST_PATH = PROJECT_ROOT / "experiments" / "pipeline3_5000" / "dataset_manifest.json"
FINAL_CONFIG_PATH = PROJECT_ROOT / "experiments" / "pipeline3_5000" / "final" / "config.json"
EXPECTED_DATASET_FINGERPRINT = "7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625"
EXPECTED_SPLIT_FINGERPRINT = "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"
EXPECTED_CHECKPOINT_SHA256 = "0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f"
EXPECTED_STATE_SHA256 = "85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634"
EXPECTED_CROMA_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"
CROMA_RECEIPT_PATH = PROJECT_ROOT / "experiments" / "pass5" / "artifacts" / "receipt.json"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class IdentityResolutionError(ValueError):
    """An authoritative dataset identity resolved to zero or multiple bundles."""


class ModelContractError(ValueError):
    """A frozen representation or checkpoint contract was violated."""


class Pipeline3ManifestResolver:
    """Resolve exact full identities from the frozen 5,000-area manifest."""

    def __init__(self, manifest_path: Path = MANIFEST_PATH, cache_root: Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        rows = self.manifest.get("rows")
        if not isinstance(rows, list) or len(rows) != 5000:
            raise IdentityResolutionError("Pipeline 3 manifest must contain exactly 5,000 rows")
        self.rows = tuple(rows)
        self._indices = {
            field: self._build_index(field) for field in ("area_id", "s2_identity", "s1_identity")
        }
        configured = json.loads(FINAL_CONFIG_PATH.read_text(encoding="utf-8"))
        checkpoint = Path(configured["checkpoint"])
        self.cache_root = Path(cache_root or checkpoint.parents[2] / "scene_features")
        self.s1_archive = self.cache_root / "source_cache" / "BigEarthNet-S1-selected.zip"
        self.reference_archive = self.cache_root / "source_cache" / "Reference_Maps-selected.zip"

    def _build_index(self, field: str) -> dict[str, tuple[Mapping[str, Any], ...]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in self.rows:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise IdentityResolutionError(f"Manifest row lacks authoritative {field}")
            grouped.setdefault(value, []).append(row)
        return {key: tuple(value) for key, value in grouped.items()}

    def resolve_row(self, identity: str) -> Mapping[str, Any]:
        """Use only exact authoritative identities; ambiguity always fails."""
        if not isinstance(identity, str) or not identity.strip():
            raise IdentityResolutionError("An exact non-empty area identity is required")
        matches: list[Mapping[str, Any]] = []
        matched_fields: list[str] = []
        for field, index in self._indices.items():
            current = list(index.get(identity, ()))
            if current:
                matched_fields.append(field)
                matches.extend(current)
        unique = {str(row["area_id"]): row for row in matches}
        if not unique:
            raise IdentityResolutionError(
                f"No exact Pipeline 3 area matches {identity!r}; full area_id, s2_identity, or s1_identity is required"
            )
        if len(unique) != 1:
            raise IdentityResolutionError(
                f"Ambiguous Pipeline 3 identity {identity!r}: matched {len(unique)} areas via {matched_fields}"
            )
        row = next(iter(unique.values()))
        for field in ("area_id", "s2_identity", "s1_identity"):
            if len(self._indices[field].get(str(row[field]), ())) != 1:
                raise IdentityResolutionError(f"Resolved row has non-unique authoritative {field}: {row[field]}")
        return row

    @staticmethod
    def _one_member(archive: zipfile.ZipFile, suffix: str, role: str) -> str:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise IdentityResolutionError(f"Expected exactly one {role} member ending {suffix!r}; found {len(matches)}")
        return matches[0]

    def resolve_bundle(self, identity: str) -> dict[str, Any]:
        row = self.resolve_row(identity)
        area_id = str(row["area_id"])
        tile = area_id.rsplit("_", 2)[0]
        optical: dict[str, Path] = {}
        for band in OPTICAL_BANDS:
            candidates = []
            for root_value in row.get("s2_source_roots", []):
                candidate = Path(root_value) / tile / area_id / f"{area_id}_{band}.tif"
                if candidate.is_file():
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise IdentityResolutionError(
                    f"{area_id}: expected exactly one S2 {band} raster from manifest roots; found {len(candidates)}"
                )
            optical[band] = candidates[0]
        if not self.s1_archive.is_file():
            raise IdentityResolutionError(f"Missing authoritative S1 archive: {self.s1_archive}")
        if not self.reference_archive.is_file():
            raise IdentityResolutionError(f"Missing authoritative reference archive: {self.reference_archive}")
        with zipfile.ZipFile(self.s1_archive) as archive:
            s1 = {
                band: self._one_member(archive, f"/{row['s1_identity']}/{row['s1_identity']}_{band}.tif", f"S1 {band}")
                for band in SAR_BANDS
            }
        with zipfile.ZipFile(self.reference_archive) as archive:
            reference = self._one_member(
                archive, f"/{area_id}/{area_id}_reference_map.tif", "reference map"
            )
        return {"row": row, "optical": optical, "sar_members": s1, "reference_member": reference}


def list_pipeline3_area_ids() -> list[str]:
    return sorted(str(row["area_id"]) for row in Pipeline3ManifestResolver().rows)


def _profile(source) -> dict[str, Any]:
    return {"crs": source.crs, "transform": source.transform, "bounds": tuple(source.bounds),
            "width": source.width, "height": source.height}


def _read_open(source, grid: Mapping[str, Any], resampling: Resampling, role: str) -> np.ndarray:
    if source.count != 1 or source.crs is None or source.transform.a <= 0 or source.transform.e >= 0:
        raise IdentityResolutionError(f"Invalid {role} raster profile")
    if source.crs != grid["crs"] or not np.allclose(tuple(source.bounds), grid["bounds"], atol=1e-6, rtol=0):
        raise IdentityResolutionError(f"{role} CRS/extent does not match the exact area bundle")
    values = source.read(1).astype(np.float32)
    if not np.all(source.read_masks(1) == 255) or not np.isfinite(values).all():
        raise IdentityResolutionError(f"{role} contains invalid, masked, or non-finite pixels")
    if source.width == grid["width"] and source.height == grid["height"] and source.transform == grid["transform"]:
        return values
    destination = np.empty((grid["height"], grid["width"]), dtype=np.float32)
    reproject(values, destination, src_transform=source.transform, src_crs=source.crs,
              dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=resampling)
    if not np.isfinite(destination).all():
        raise IdentityResolutionError(f"{role} became non-finite on the canonical grid")
    return destination


def _read_path(path: Path, grid: Mapping[str, Any], role: str) -> np.ndarray:
    with rasterio.open(path) as source:
        return _read_open(source, grid, Resampling.bilinear, role)


def _read_bytes(blob: bytes, grid: Mapping[str, Any], resampling: Resampling, role: str) -> np.ndarray:
    with MemoryFile(blob) as memory, memory.open() as source:
        return _read_open(source, grid, resampling, role)


@lru_cache(maxsize=1)
def _load_probe(checkpoint_text: str) -> tuple[ProbeHead, str, str]:
    checkpoint = Path(checkpoint_text)
    file_hash = _sha256_path(checkpoint)
    if file_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ModelContractError(f"Validated checkpoint file hash mismatch: {file_hash}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("input_dimension") != 830 or payload.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
        raise ModelContractError("Validated checkpoint metadata does not match the frozen 830D dataset contract")
    model = ProbeHead(830).eval()
    model.load_state_dict(payload["state_dict"])
    state_hash = _state_sha256(model)
    if state_hash != EXPECTED_STATE_SHA256:
        raise ModelContractError(f"Validated checkpoint state hash mismatch: {state_hash}")
    return model, file_hash, state_hash


@lru_cache(maxsize=1)
def _load_croma(source_text: str, checkpoint_text: str) -> CROMAAdapter:
    checkpoint = Path(checkpoint_text)
    digest = _sha256_path(checkpoint)
    if digest != EXPECTED_CROMA_SHA256:
        raise ModelContractError(f"Official CROMA checkpoint hash mismatch: {digest}")
    return CROMAAdapter(Path(source_text), checkpoint)


def _croma_assets() -> tuple[Path, Path]:
    """Use explicit settings, then the existing audited provenance receipt."""
    settings = get_settings()
    if (settings.croma_source / "use_croma.py").is_file() and settings.croma_checkpoint.is_file():
        return settings.croma_source, settings.croma_checkpoint
    receipt = json.loads(CROMA_RECEIPT_PATH.read_text(encoding="utf-8"))
    provenance = receipt.get("provenance") or {}
    source = Path(provenance.get("croma_source", ""))
    checkpoint = Path(provenance.get("croma_checkpoint", ""))
    if not (source / "use_croma.py").is_file() or not checkpoint.is_file():
        raise ModelContractError("Official CROMA assets are unavailable in settings and audited provenance")
    if provenance.get("croma_checkpoint_sha256") != EXPECTED_CROMA_SHA256:
        raise ModelContractError("Audited CROMA provenance does not match the frozen checkpoint")
    return source, checkpoint


def _public_features(physical_report: Mapping[str, Any], deep: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    representations = {key: list(value.shape) for key, value in deep.items()}
    return {
        "status": "computed",
        "spectral": {"status": "computed", "source": "frozen Pipeline 3 physical feature extraction",
                     "schema": "pipeline3_physical_62D", **dict(physical_report),
                     "statistics": dict(physical_report["feature_values"])},
        "deep": {"status": "computed", "source": "official CROMA base",
                 "representations": representations, "pooled_dimension": 768,
                 "token_semantics": "latent representations only; not 225x19 predictions"},
        "hybrid": {"status": "computed", "source": "physical_62D + joint_croma_gap_768D",
                   "dimension": 830, "components": {"physical": 62, "joint_croma_gap": 768}},
    }


def validate_representation_dimensions(physical: np.ndarray, deep: Mapping[str, torch.Tensor],
                                       hybrid: np.ndarray | None = None,
                                       prediction: np.ndarray | None = None) -> None:
    if tuple(physical.shape) != (62,) or not np.isfinite(physical).all():
        raise ModelContractError(f"Expected physical_62D, got {physical.shape}")
    required_shapes = {
        "optical_encodings": (1, 225, 768), "SAR_encodings": (1, 225, 768),
        "joint_encodings": (1, 225, 768), "joint_GAP": (1, 768),
    }
    for key, shape in required_shapes.items():
        value = deep.get(key)
        actual = tuple(value.shape) if value is not None else None
        if actual != shape or not torch.isfinite(value).all():
            raise ModelContractError(f"Invalid {key} dimensions: expected {shape}, got {actual}")
    if hybrid is not None and (hybrid.shape != (830,) or not np.isfinite(hybrid).all()):
        raise ModelContractError(f"Expected hybrid_830D, got {hybrid.shape}")
    if prediction is not None and (prediction.shape != (19,) or not np.isfinite(prediction).all()):
        raise ModelContractError(f"Expected scene_prediction_19D, got {prediction.shape}")


def run_pipeline3_scene_probe(request: Mapping[str, Any], *, resolver: Pipeline3ManifestResolver | None = None,
                              checkpoint: Path | None = None) -> dict[str, Any]:
    """Run exact-area inference and return the legacy-compatible analysis report."""
    started = time.perf_counter()
    identity = request.get("sample_id")
    if not identity:
        raise IdentityResolutionError("The validated scene probe requires an exact sample_id")
    resolver = resolver or Pipeline3ManifestResolver()
    bundle = resolver.resolve_bundle(str(identity))
    row = bundle["row"]
    area_id = str(row["area_id"])
    expected_hashes = row["content_sha256"]

    b02 = bundle["optical"]["B02"]
    with rasterio.open(b02) as source:
        grid = _profile(source)
    if (grid["height"], grid["width"]) != (120, 120):
        raise IdentityResolutionError(f"{area_id}: canonical B02 grid is not 120x120")
    optical_values = []
    for band in OPTICAL_BANDS:
        path = bundle["optical"][band]
        digest = _sha256_path(path)
        if digest != expected_hashes["s2_bands"][band]:
            raise IdentityResolutionError(f"{area_id}: S2 {band} content hash mismatch")
        optical_values.append(_read_path(path, grid, f"S2 {band}"))
    with zipfile.ZipFile(resolver.s1_archive) as archive:
        sar_blobs = {band: archive.read(bundle["sar_members"][band]) for band in SAR_BANDS}
    for band in SAR_BANDS:
        if _sha256_bytes(sar_blobs[band]) != expected_hashes[f"s1_{band.lower()}"]:
            raise IdentityResolutionError(f"{area_id}: S1 {band} content hash mismatch")
    sar = np.stack([_read_bytes(sar_blobs[band], grid, Resampling.bilinear, f"S1 {band}") for band in SAR_BANDS])
    with zipfile.ZipFile(resolver.reference_archive) as archive:
        reference_blob = archive.read(bundle["reference_member"])
    if _sha256_bytes(reference_blob) != expected_hashes["reference"]:
        raise IdentityResolutionError(f"{area_id}: reference content hash mismatch")
    reference = _read_bytes(reference_blob, grid, Resampling.nearest, "reference map")
    if not np.all(reference == np.floor(reference)):
        raise IdentityResolutionError(f"{area_id}: reference map contains non-integer codes")
    optical = np.stack(optical_values)

    physical, physical_report = LocalRasterFeatureProvider().extract(
        optical, sar, list(OPTICAL_BANDS), list(SAR_BANDS)
    )
    croma_source, croma_checkpoint = _croma_assets()
    adapter = _load_croma(str(croma_source), str(croma_checkpoint))
    deep = adapter.infer(optical, sar)
    validate_representation_dimensions(physical, deep)
    joint_gap = deep["joint_GAP"].detach().cpu().to(torch.float32).numpy()[0]
    hybrid = np.concatenate((physical.astype(np.float32), joint_gap))
    validate_representation_dimensions(physical, deep, hybrid)

    configured = json.loads(FINAL_CONFIG_PATH.read_text(encoding="utf-8"))
    checkpoint_path = Path(checkpoint or configured["checkpoint"])
    model, file_hash, state_hash = _load_probe(str(checkpoint_path))
    with torch.inference_mode():
        prediction = model.predict(torch.from_numpy(hybrid).unsqueeze(0))[0].numpy()
    validate_representation_dimensions(physical, deep, hybrid, prediction)

    analysis_id = str(uuid4())
    croma_evidence = {
        mode: {"representation": f"{mode}_encodings", "shape": [1, 225, 768],
               "semantic_status": "latent; no assigned class meaning"}
        for mode in ("optical", "sar", "joint")
    }
    evidence = build_spatial_evidence(
        sample_id=area_id, optical=optical, sar=sar, transform=list(grid["transform"]), crs=str(grid["crs"]),
        scene_metadata={"dataset": resolver.manifest.get("dataset"), "split": row["split"],
                        "s1_identity": row["s1_identity"], "s2_identity": row["s2_identity"]},
        provenance={"analysis_id": analysis_id, "retrieval_source": "exact Pipeline 3 manifest bundle",
                    "pixel_source_artifact": f"analysis:{analysis_id}:validated_exact_rasters",
                    "method": "deterministic 15x15 token aggregation; independent of scene probe"},
        croma=croma_evidence,
    )
    simple = interpret_evidence(evidence, str(request["query"]), mode="simple")
    technical = technical_companion(evidence, str(request["query"]))
    validate_interpretation(simple)
    validate_interpretation(technical)
    features = _public_features(physical_report, deep)
    model_output = {
        "status": "available", "prediction": {"level": "scene", "classes": 19,
            "shape": [19], "vector": prediction.astype(float).tolist(),
            "class_order": [name for name, _ in CLASSES],
            "model": "validated 830D hybrid linear probe",
            "input": {"physical_62D": 62, "joint_croma_gap_768D": 768, "hybrid_830D": 830},
            "checkpoint": {"file_sha256": file_hash, "state_dict_sha256": state_hash},
            "calibration_status": "unavailable", "semantic_interpretation": "not_semantically_interpreted",
            "spatial_prediction": False,
        }
    }
    public_spatial = {key: evidence[key] for key in ("schema_version", "scene", "sensor_views", "claims", "regions", "provenance")}
    trace = [
        {"operation": "Exact manifest identity resolved", "status": "completed", "detail": area_id},
        {"operation": "S1/S2/reference hashes and grids validated", "status": "completed"},
        {"operation": "physical_62D", "status": "completed", "dimensions": [62]},
        {"operation": "optical/SAR/joint CROMA tokens", "status": "completed", "dimensions": [1, 225, 768]},
        {"operation": "joint CROMA GAP", "status": "completed", "dimensions": [1, 768]},
        {"operation": "hybrid_830D", "status": "completed", "dimensions": [830]},
        {"operation": "scene probe", "status": "completed", "dimensions": [19], "spatial": False},
        {"operation": "Deterministic evidence and controlled interpretation", "status": "completed"},
    ]
    for item in trace:
        item["message"] = item["operation"]
    return {
        "schema_version": "pipeline3_scene_probe_v1", "analysis_id": analysis_id, "status": "completed",
        "timestamp": time.time(),
        "query_interpretation": {"task": "deterministic_scene_analysis", "modalities": ["optical", "sar"]},
        "data_retrieval": {"source": "exact Pipeline 3 manifest bundle", "assets": {
            "dataset": str(resolver.manifest.get("dataset")), "area_id": area_id,
            "sample_id": area_id, "s1_identity": str(row["s1_identity"]), "s2_identity": str(row["s2_identity"]),
            "split": str(row["split"])}, "exact_match_count": 1},
        "data_cube": {"modalities": ["optical", "sar"], "spatial_dimensions": [120, 120], "metadata": {
            "optical": {"crs": str(grid["crs"]), "bounds": list(grid["bounds"]),
                        "resolution": [abs(grid["transform"].a), abs(grid["transform"].e)],
                        "transform": list(grid["transform"]), "shape": [12, 120, 120]},
            "sar": {"crs": str(grid["crs"]), "bounds": list(grid["bounds"]),
                    "resolution": [abs(grid["transform"].a), abs(grid["transform"].e)],
                    "transform": list(grid["transform"]), "shape": [2, 120, 120]}}},
        "validation": {"valid": True, "errors": [], "warnings": [], "checks": {
            "exact_identity": True, "exact_sensor_identity": True, "complete_raster_bundle": True,
            "content_hashes": True, "optical_sar_compatibility": {"crs": True, "bounds": True, "grid": True}}},
        "preprocessing": [{"operation": "canonical B02 120x120 grid", "status": "completed"},
                          {"operation": "bilinear S2/S1 reprojection; nearest reference validation", "status": "completed"},
                          {"operation": "official CROMA per-channel normalization", "status": "completed"}],
        "features": features, "model_results": model_output,
        "spatial_evidence": {"status": "AVAILABLE", **public_spatial},
        "interpretation": {**simple, "technical_answer": technical["answer"], "technical_claims": technical["claims"],
                           "model_output_semantics": "not_semantically_interpreted"},
        "evidence": [{"kind": "scene model output", "reference": "model_results.prediction", "status": "available"},
                     {"kind": "deterministic spatial evidence", "reference": "spatial_evidence", "status": "available"}],
        "confidence": {"prediction_status": "scene-level output available", "calibration_status": "unavailable",
                       "accuracy_status": "reproduced test aggregate 65.0%; not per-scene confidence",
                       "validation_status": "passed"},
        "warnings": ["The 19D output is scene-level and uncalibrated; it is not a token map, segmentation, or grounding.",
                     "CROMA token tensors are latent representations [225,768], not 225x19 predictions."],
        "llm_explanation": "Validated scene-level 830D-to-19 probe output is available separately from deterministic evidence and controlled interpretation.",
        "execution_trace": trace, "device": str(adapter.device), "runtime_seconds": time.perf_counter() - started,
        "scientific_provenance": {"dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT, "split": row["split"],
            "area_id": area_id, "s1_identity": row["s1_identity"], "s2_identity": row["s2_identity"],
            "checkpoint_file_sha256": file_hash, "checkpoint_state_sha256": state_hash},
    }
