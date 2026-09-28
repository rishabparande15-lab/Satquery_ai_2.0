"""Authoritative per-image representation availability catalog.

The catalog is an inventory and validation boundary. It never infers
representations from filenames or filesystem location and never fabricates
receipt metadata.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .representation_artifacts import ArtifactResolver, file_sha256

SCHEMA_VERSION = "representation_catalog_v1"
STATUSES = frozenset({"VERIFIED", "MISSING", "INVALID", "EXPIRED", "PROVENANCE_MISMATCH", "SPLIT_MISMATCH"})
DEFAULT_TYPES = (
    "raw_optical", "raw_sar", "physical_62d", "optical_croma_tokens", "sar_croma_tokens",
    "joint_croma_tokens", "optical_gap", "sar_gap", "joint_croma_gap_768d", "pooled_croma",
    "hybrid_830d", "pixel_features", "token_features", "regions", "evidence",
)


@dataclass(frozen=True)
class CatalogRecord:
    image_id: str
    split: str
    representation_type: str
    representation_id: str | None
    artifact_ref: Mapping[str, Any] | None
    logical_uri: str | None
    checksum: str | None
    size: int | None
    shape: tuple[int, ...] | None
    dtype: str | None
    modality: str | None
    spatial_semantics: Mapping[str, Any] | None
    producer: str | None
    model: str | None
    model_version: str | None
    preprocessing_version: str | None
    dataset_version: str | None
    provenance: Mapping[str, Any]
    availability: str
    validation_status: str
    validation_issues: tuple[str, ...] = ()
    shard_ref: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.image_id or not self.split or not self.representation_type:
            raise ValueError("catalog identity fields are required")
        if self.availability not in STATUSES or self.validation_status not in STATUSES:
            raise ValueError("unsupported catalog status")
        if self.shape is not None and any(type(value) is not int or value <= 0 for value in self.shape):
            raise ValueError("shape must contain positive integers")
        if self.checksum is not None and (len(self.checksum) != 64 or any(c not in "0123456789abcdef" for c in self.checksum.lower())):
            raise ValueError("checksum must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset manifest has no rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = row.get("area_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("manifest row has no area_id")
        if image_id in result:
            raise ValueError(f"duplicate manifest image: {image_id}")
        result[image_id] = {"image_id": image_id, "split": row.get("split"),
                    "dataset_version": payload.get("dataset"),
                    "dataset_fingerprint": payload.get("dataset_fingerprint")}
    if any(row["split"] not in {"train", "validation", "test"} for row in result.values()):
        raise ValueError("manifest row has an invalid split")
    return result


def records_from_receipt(
    receipt_path: Path,
    *,
    manifest: Mapping[str, Mapping[str, Any]],
    resolver: ArtifactResolver | None = None,
) -> list[CatalogRecord]:
    """Convert a receipt into verified records, requiring image/split identity."""
    receipt_path = Path(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    image_id = payload.get("scene_id", payload.get("sample_id"))
    if not isinstance(image_id, str):
        raise ValueError("receipt has no scene/sample identity")
    manifest_row = manifest.get(image_id)
    split = payload.get("split")
    if manifest_row is None:
        status, issues = "PROVENANCE_MISMATCH", ("image_outside_manifest",)
        expected_split = split or "UNKNOWN"
    else:
        expected_split = manifest_row["split"]
        status, issues = "VERIFIED", ()
        if split is not None and split != expected_split:
            status, issues = "SPLIT_MISMATCH", ("receipt_split_disagrees_with_manifest",)
    arrays = payload.get("arrays") or payload.get("representations")
    if not isinstance(arrays, Mapping):
        return []
    result = []
    for name, metadata in sorted(arrays.items()):
        if not isinstance(metadata, Mapping):
            continue
        filename = metadata.get("file")
        checksum = metadata.get("sha256")
        local_path = receipt_path.parent / filename if isinstance(filename, str) else None
        record_status, record_issues = status, list(issues)
        if record_status == "VERIFIED":
            if not isinstance(checksum, str) or local_path is None or not local_path.is_file():
                record_status, record_issues = "INVALID", ["receipt_or_artifact_missing"]
            elif file_sha256(local_path) != checksum:
                record_status, record_issues = "INVALID", ["artifact_checksum_mismatch"]
        result.append(CatalogRecord(
            image_id=image_id, split=expected_split, representation_type=_type_for(name),
            representation_id=f"{image_id}:{name}:{checksum[:16] if isinstance(checksum, str) else 'unverified'}",
            artifact_ref={"receipt_file": receipt_path.name, "file": filename} if filename else None,
            logical_uri=None, checksum=checksum if isinstance(checksum, str) else None,
            size=local_path.stat().st_size if local_path and local_path.is_file() else None,
            shape=tuple(metadata["shape"]) if isinstance(metadata.get("shape"), list) else None,
            dtype=metadata.get("dtype") if isinstance(metadata.get("dtype"), str) else None,
            modality=_modality_for(name), spatial_semantics={"level": _spatial_level_for(name)},
            producer=(payload.get("provenance") or {}).get("producer"), model=None,
            model_version=(payload.get("provenance") or {}).get("model_version"),
            preprocessing_version=(payload.get("provenance") or {}).get("preprocessing_version"),
            dataset_version=manifest_row.get("dataset_version") if manifest_row else None,
            provenance={"receipt_sha256": file_sha256(receipt_path), "receipt_file": receipt_path.name},
            availability=record_status, validation_status=record_status,
            validation_issues=tuple(sorted(set(record_issues))),
        ))
    return result


def records_from_shard_manifest(
    cache_manifest_path: Path,
    *,
    manifest: Mapping[str, Mapping[str, Any]],
    artifact_namespace: str = "pipeline3-5000",
) -> list[CatalogRecord]:
    """Admit only verified rows from the existing Pipeline 3 feature-cache format."""
    cache_manifest_path = Path(cache_manifest_path)
    payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("sample_count") != len(manifest):
        raise ValueError("feature cache is not a complete authoritative population")
    manifest_fingerprints = {row.get("dataset_fingerprint") for row in manifest.values() if row.get("dataset_fingerprint")}
    if manifest_fingerprints and payload.get("dataset_fingerprint") not in manifest_fingerprints:
        raise ValueError("feature cache dataset fingerprint disagrees with image manifest")
    if not payload.get("dataset_fingerprint"):
        raise ValueError("feature cache lacks dataset fingerprint")
    rows: list[CatalogRecord] = []
    seen: set[tuple[str, str]] = set()
    cache_root = cache_manifest_path.parent
    for shard_entry in payload.get("shards", []):
        shard_path = Path(shard_entry.get("path", ""))
        if not shard_path.is_absolute():
            shard_path = cache_root / shard_path
        if not shard_path.is_file() or file_sha256(shard_path) != shard_entry.get("sha256"):
            raise ValueError(f"feature shard checksum or path mismatch: {shard_path}")
        import numpy as np
        with np.load(shard_path, allow_pickle=False) as shard:
            required = {"area_ids", "splits", "physical", "optical_croma", "sar_croma", "joint_croma"}
            if not required.issubset(shard.files):
                raise ValueError("feature shard lacks required representation arrays")
            area_ids = shard["area_ids"].tolist()
            splits = shard["splits"].tolist()
            if len(area_ids) != shard_entry.get("rows") or len(area_ids) != len(splits):
                raise ValueError("feature shard row count mismatch")
            type_arrays = {
                "physical_features": shard["physical"],
                "optical_croma_gap": shard["optical_croma"],
                "sar_croma_gap": shard["sar_croma"],
                "joint_croma_gap": shard["joint_croma"],
            }
            for index, image_id in enumerate(area_ids):
                if image_id not in manifest:
                    raise ValueError(f"feature shard image is outside manifest: {image_id}")
                if splits[index] != manifest[image_id]["split"]:
                    raise ValueError(f"feature shard split mismatch: {image_id}")
                for representation_type, values in type_arrays.items():
                    key = (image_id, representation_type)
                    if key in seen:
                        raise ValueError(f"duplicate shard representation: {key}")
                    seen.add(key)
                    sample = values[index]
                    rows.append(CatalogRecord(
                        image_id=image_id, split=splits[index], representation_type=representation_type,
                        representation_id=f"{image_id}:{representation_type}:shard:{index}",
                        artifact_ref={"artifact_checksum": shard_entry["sha256"], "sample_index": index,
                                      "shard_rows": len(area_ids)},
                        logical_uri=f"artifact://{artifact_namespace}/shards/{shard_path.name}",
                        checksum=shard_entry["sha256"], size=shard_path.stat().st_size,
                        shape=tuple(sample.shape), dtype=str(sample.dtype), modality=_modality_for(representation_type),
                        spatial_semantics={"level": "scene"}, producer="run_pipeline3_5000_baseline",
                        model="official CROMA" if "croma" in representation_type else None,
                        model_version=payload.get("croma_checkpoint_sha256"),
                        preprocessing_version=payload.get("normalization_implementation"),
                        dataset_version=payload.get("dataset_fingerprint"),
                        provenance={"cache_manifest_sha256": file_sha256(cache_manifest_path),
                                    "shard_checksum_is_artifact_checksum": True},
                        availability="VERIFIED", validation_status="VERIFIED",
                        shard_ref={"shard_name": shard_path.name, "sample_index": index,
                                   "sample_identity_field": "area_ids"},
                    ))
    if len({record.image_id for record in rows}) != len(manifest):
        raise ValueError("feature cache does not cover every manifest image")
    return rows


def records_from_materialization_manifest(
    materialization_manifest_path: Path,
    *,
    manifest: Mapping[str, Mapping[str, Any]],
    artifact_namespace: str = "pipeline3-5000",
) -> list[CatalogRecord]:
    """Admit Phase 2D.2 typed shard receipts after full integrity validation."""
    from .representation_materialization import DATASET_FINGERPRINT
    from .receipt_catalog import validate_materialization_receipts

    receipts = validate_materialization_receipts(materialization_manifest_path)
    rows: list[CatalogRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for receipt in receipts:
        image_id = receipt["image_id"]
        representation_type = receipt["representation_type"]
        variant = receipt["variant"]
        identity = (image_id, representation_type, variant)
        if identity in seen:
            raise ValueError(f"duplicate materialized representation: {identity}")
        seen.add(identity)
        expected = manifest.get(image_id)
        if expected is None:
            raise ValueError(f"materialized image is outside manifest: {image_id}")
        if receipt["split"] != expected["split"]:
            raise ValueError(f"materialized split mismatch: {image_id}")
        if receipt["dataset_fingerprint"] != DATASET_FINGERPRINT:
            raise ValueError(f"materialized provenance mismatch: {image_id}")
        path = Path(receipt["artifact_path"])
        checksum = receipt["artifact_sha256"]
        rows.append(CatalogRecord(
            image_id=image_id, split=receipt["split"], representation_type=representation_type,
            representation_id=f"{image_id}:{representation_type}:{variant}:{receipt['sample_sha256'][:16]}",
            artifact_ref={"artifact_checksum": checksum, "array_key": receipt["array_key"],
                          "sample_index": receipt["sample_index"], "sample_sha256": receipt["sample_sha256"],
                          "variant": variant},
            logical_uri=f"artifact://{artifact_namespace}/shards/{receipt['shard_id']}/{representation_type}",
            checksum=checksum, size=path.stat().st_size, shape=tuple(receipt["shape"]), dtype=receipt["dtype"],
            modality=_modality_for(representation_type), spatial_semantics={"level": "scene"},
            producer=receipt["producer"], model=receipt.get("model"), model_version=receipt.get("model_version"),
            preprocessing_version=receipt.get("preprocessing_version"), dataset_version=DATASET_FINGERPRINT,
            provenance={"sample_sha256": receipt["sample_sha256"], "variant": variant,
                        "receipt_schema": receipt["schema_version"],
                        "shard_checksum_is_artifact_checksum": True},
            availability="VERIFIED", validation_status="VERIFIED",
            shard_ref={"shard_id": receipt["shard_id"], "sample_index": receipt["sample_index"],
                       "sample_identity_field": "ordered_image_ids", "array_key": receipt["array_key"]},
        ))
    expected_count = len(manifest) * 3
    if len(rows) != expected_count:
        raise ValueError(f"materialized core receipt count is {len(rows)}, expected {expected_count}")
    return rows


def build_catalog(
    manifest_path: Path,
    *,
    receipt_paths: Iterable[Path] = (),
    shard_manifest_paths: Iterable[Path] = (),
    materialization_manifest_paths: Iterable[Path] = (),
    representation_types: Iterable[str] = DEFAULT_TYPES,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    types = tuple(sorted(set(representation_types)))
    records = []
    discovered: dict[tuple[str, str], CatalogRecord] = {}
    receipt_records: list[CatalogRecord] = []
    shard_records: list[CatalogRecord] = []
    materialization_records: list[CatalogRecord] = []
    for receipt_path in receipt_paths:
        parsed_records = records_from_receipt(receipt_path, manifest=manifest)
        receipt_records.extend(parsed_records)
        for record in parsed_records:
            discovered[(record.image_id, record.representation_type)] = record
    for shard_manifest_path in shard_manifest_paths:
        parsed_records = records_from_shard_manifest(shard_manifest_path, manifest=manifest)
        shard_records.extend(parsed_records)
        for record in parsed_records:
            key = (record.image_id, record.representation_type)
            if key in discovered:
                raise ValueError(f"duplicate image/type representation admission: {key}")
            discovered[key] = record
    for materialization_manifest_path in materialization_manifest_paths:
        parsed_records = records_from_materialization_manifest(materialization_manifest_path, manifest=manifest)
        materialization_records.extend(parsed_records)
        for record in parsed_records:
            key = (record.image_id, record.representation_type)
            if key in discovered:
                raise ValueError(f"duplicate image/type representation admission: {key}")
            discovered[key] = record
    for image_id, row in sorted(manifest.items()):
        for representation_type in types:
            record = discovered.get((image_id, representation_type))
            if record is None:
                record = CatalogRecord(
                    image_id=image_id, split=row["split"], representation_type=representation_type,
                    representation_id=None, artifact_ref=None, logical_uri=None, checksum=None, size=None,
                    shape=None, dtype=None, modality=_modality_for(representation_type),
                    spatial_semantics={"level": _spatial_level_for(representation_type)}, producer=None, model=None,
                    model_version=None, preprocessing_version=None, dataset_version=row["dataset_version"],
                    provenance={}, availability="MISSING", validation_status="MISSING",
                    validation_issues=("no_verified_receipt",),
                )
            records.append(record)
    records.sort(key=lambda record: (record.image_id, record.representation_type))
    canonical = b"".join(_canonical_bytes(record.to_dict()) + b"\n" for record in records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "manifest_image_count": len(manifest),
        "represented_image_count": len({record.image_id for record in records if record.availability == "VERIFIED"}),
        "unrepresented_image_count": len(manifest) - len({record.image_id for record in records if record.availability == "VERIFIED"}),
        "record_count": len(records),
        "counts_by_status": dict(sorted(Counter(record.availability for record in records).items())),
        "counts_by_split": dict(sorted(Counter(record.split for record in records if record.availability == "VERIFIED").items())),
        "counts_by_type": dict(sorted(Counter(record.representation_type for record in records if record.availability == "VERIFIED").items())),
        "counts_by_split_and_type": _compound_counts(record for record in records if record.availability == "VERIFIED"),
        "catalog_rows_by_split": dict(sorted(Counter(record.split for record in records).items())),
        "catalog_rows_by_type": dict(sorted(Counter(record.representation_type for record in records).items())),
        "catalog_rows_by_split_and_status": _compound_counts(record for record in records),
        "receipt_records_inspected": len(receipt_records),
        "receipt_status_counts": dict(sorted(Counter(record.availability for record in receipt_records).items())),
        "receipt_images_outside_manifest": sorted({record.image_id for record in receipt_records if record.image_id not in manifest}),
        "shard_records_verified": len(shard_records),
        "shard_images_verified": len({record.image_id for record in shard_records}),
        "shard_backed_types": dict(sorted(Counter(record.representation_type for record in shard_records).items())),
        "materialization_records_verified": len(materialization_records),
        "complete_core_image_count": sum(
            all((image_id, name) in discovered and discovered[(image_id, name)].availability == "VERIFIED"
                for name in ("physical_62d", "joint_croma_gap_768d", "hybrid_830d"))
            for image_id in manifest
        ),
        "catalog_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    return {"schema_version": SCHEMA_VERSION, "records": records, "report": report, "canonical_bytes": canonical}


def write_catalog(catalog: Mapping[str, Any], output: Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = catalog["records"]
    content = catalog["canonical_bytes"]
    (output / "records.jsonl").write_bytes(content)
    manifest = {"schema_version": SCHEMA_VERSION, "record_count": len(records), "records_sha256": hashlib.sha256(content).hexdigest()}
    (output / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
    (output / "validation_report.json").write_bytes(_canonical_bytes(catalog["report"]) + b"\n")
    (output / "provenance.json").write_bytes(_canonical_bytes({"schema_version": SCHEMA_VERSION, "machine_paths_excluded": True, "receipt_validation": "reuse_existing_sha256_logic"}) + b"\n")
    return catalog["report"]


def _type_for(name: str) -> str:
    mapping = {"croma_optical_encodings": "optical_croma_tokens", "croma_SAR_encodings": "sar_croma_tokens", "croma_joint_encodings": "joint_croma_tokens", "croma_optical_GAP": "optical_croma_gap", "croma_SAR_GAP": "sar_croma_gap", "croma_joint_GAP": "joint_croma_gap", "physical_features": "physical_features", "hybrid_features": "hybrid", "pooled_croma_features": "pooled_croma"}
    return mapping.get(name, name)


def _modality_for(name: str) -> str | None:
    if "optical" in name.lower() or name in {"B04", "B08", "NDVI", "NDWI", "MNDWI", "NDBI", "BSI"}:
        return "optical"
    if "sar" in name.lower() or name in {"VV", "VH", "VV_minus_VH"}:
        return "sar"
    if "joint" in name.lower() or name in {
        "hybrid", "hybrid_830d", "physical_features", "physical_62d", "pooled_croma"
    }:
        return "optical_sar"
    return None


def _spatial_level_for(name: str) -> str:
    lowered = name.lower()
    if "token" in lowered or lowered.startswith("token_") or "encoding" in lowered:
        return "token"
    if "gap" in lowered or lowered in {"hybrid", "pooled_croma", "physical_features"}:
        return "scene"
    return "pixel"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _compound_counts(records: Iterable[CatalogRecord]) -> dict[str, int]:
    return dict(sorted(Counter(f"{record.split}|{record.representation_type}|{record.availability}" for record in records).items()))
