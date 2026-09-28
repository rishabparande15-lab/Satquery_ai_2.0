"""Raw-S2 recovery and fail-closed input gate for Phase 3O.1."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import rasterio

from ..eo_vlm_adapter import S2_BANDS

SAFE_TASKS = frozenset({"binary_qa", "multiple_choice_qa", "caption"})
EXPECTED_NATIVE = {"B01": (20, 20), "B02": (120, 120), "B03": (120, 120), "B04": (120, 120),
                   "B05": (60, 60), "B06": (60, 60), "B07": (60, 60), "B08": (120, 120),
                   "B8A": (60, 60), "B09": (20, 20), "B11": (60, 60), "B12": (60, 60)}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedS2Asset:
    sample_id: str
    image_id: str
    split: str
    local_path: str
    source_revision: str
    sha256: str
    shape: tuple[int, int, int]
    dtype: str
    band_order: tuple[str, ...]
    normalization_revision: str
    provenance: Mapping[str, Any]

    def validate(self) -> None:
        if self.split not in {"train", "validation"}: raise ValueError("only train or validation split raw assets are permitted")
        if not self.sample_id or not self.image_id or not self.local_path: raise ValueError("identity and path are required")
        if not self.source_revision or len(self.sha256) != 64: raise ValueError("source revision and checksum are required")
        if self.shape != (12, 120, 120) or self.band_order != S2_BANDS: raise ValueError("invalid canonical S2 shape or band order")
        if self.normalization_revision != "robust_channel_scale_v1": raise ValueError("unknown normalization revision")
        if not self.provenance or self.provenance.get("source_record_sha256") is None: raise ValueError("complete provenance is required")

    def to_dict(self) -> dict[str, Any]: self.validate(); return asdict(self)


def validate_annotation(record: Mapping[str, Any]) -> None:
    if record.get("split") not in {"train", "validation"} or record.get("effective_partition") != record.get("split"):
        raise ValueError("test and ambiguous split records are forbidden")
    if record.get("task_type") not in SAFE_TASKS or record.get("dependency_class") != "VISUAL_ONLY":
        raise ValueError("annotation task is not permitted")
    if record.get("box") is not None or record.get("point") is not None or record.get("geometry_frame") is not None:
        raise ValueError("unverified spatial annotation is forbidden")
    if not record.get("source_revision") or not record.get("source_record_sha256"):
        raise ValueError("annotation provenance is incomplete")


def inspect_area(record: Mapping[str, Any], paths: Mapping[str, Path]) -> VerifiedS2Asset:
    validate_annotation(record)
    if tuple(paths) != S2_BANDS: raise ValueError("band ordering or membership differs from canonical S2")
    digest = sha256()
    crs = transform = bounds = None
    for band in S2_BANDS:
        path = paths[band]
        if not path.is_file(): raise ValueError("missing S2 band")
        with rasterio.open(path) as raster:
            values = raster.read(1)
            if raster.count != 1 or (raster.height, raster.width) != EXPECTED_NATIVE[band]: raise ValueError("invalid native S2 dimensions")
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all(): raise ValueError("NaN or Inf in raw S2 asset")
            if raster.nodata is not None and np.any(values == raster.nodata): raise ValueError("nodata pixels in raw S2 asset")
            if band == "B02": crs, transform, bounds = str(raster.crs), tuple(raster.transform), tuple(raster.bounds)
        file_digest = sha256_file(path); digest.update(band.encode()); digest.update(bytes.fromhex(file_digest))
    provenance = {"dataset": "BigEarthNet.txt", "source_record_sha256": record["source_record_sha256"],
                  "annotation_id": record["annotation_id"], "record_id": record["record_id"],
                  "acquisition": "existing_project_managed_pipeline3_s2", "crs": crs,
                  "affine_transform": transform, "bounds": bounds, "native_dimensions": EXPECTED_NATIVE,
                  "license_status": "PROJECT_AUTHORIZED_BIGEARTHNET_TXT",
                  "scientific_representation_used": False}
    asset = VerifiedS2Asset(record["record_id"], record["image_id"], record["split"], str(paths["B02"].parent), record["source_revision"],
                            digest.hexdigest(), (12, 120, 120), "uint16", S2_BANDS, "robust_channel_scale_v1", provenance)
    asset.validate(); return asset


def deterministic_manifest(assets: Iterable[VerifiedS2Asset], records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(assets, key=lambda item: (item.image_id, item.sample_id))
    image_ids = [item.image_id for item in values]
    if len(image_ids) != len(set(image_ids)): raise ValueError("duplicate image IDs")
    hashes = [item.sha256 for item in values]
    if len(hashes) != len(set(hashes)): raise ValueError("duplicate S2 asset hashes")
    payload = {"manifest_version": "phase3o_s2_training_manifest_v1", "dataset_revision": values[0].source_revision if values else None,
               "sample_count": len(values), "split": "train", "records": [
        {"sample_id": item.sample_id, "image_id": item.image_id, "annotation_id": records[item.sample_id]["annotation_id"],
         "task_type": records[item.sample_id]["task_type"], "s2_asset_sha256": item.sha256, "source_revision": item.source_revision,
         "representation_revision": "raw_s2_12band_pipeline3_v1", "normalization_revision": item.normalization_revision,
         "provenance": item.provenance} for item in values]}
    payload["manifest_sha256"] = sha256(canonical_bytes(payload)).hexdigest()
    return payload


def training_gate(*, assets: Iterable[VerifiedS2Asset], projector_smoke_test: str) -> dict[str, Any]:
    values = list(assets); reasons = []
    try:
        for item in values: item.validate()
        deterministic_manifest(values, {item.sample_id: {"annotation_id": item.provenance["annotation_id"], "task_type": "binary_qa"} for item in values})
    except ValueError as exc: reasons.append(str(exc))
    if not values: reasons.append("no verified raw S2 assets")
    if projector_smoke_test != "PASS": reasons.append("projector smoke test did not pass")
    ready = not reasons
    return {"training_input_ready": ready, "execution_permitted": False, "blocking_reasons": reasons,
            "sample_count": len(values), "projector_smoke_test": projector_smoke_test}
