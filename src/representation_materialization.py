"""Verified, resumable admission of the authoritative Pipeline 3 feature cache.

This module does not extract scientific features.  It verifies the frozen
Pipeline 3 cache and materializes only the deterministic 830-D concatenation
used by the evaluated hybrid checkpoint.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np

from .representation_artifacts import file_sha256


SCHEMA_VERSION = "representation_materialization_v1"
RECEIPT_SCHEMA_VERSION = "representation_shard_receipt_v1"
DATASET_FINGERPRINT = "7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625"
SPLIT_FINGERPRINT = "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"
CROMA_CHECKPOINT_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"
CROMA_SOURCE_REVISION = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
EXPECTED_SPLITS = {"train": 4600, "validation": 200, "test": 200}
CORE_TYPES = ("physical_62d", "joint_croma_gap_768d", "hybrid_830d")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sample_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_bytes(value) + b"\n")


def _load_authoritative_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    expected_total = sum(EXPECTED_SPLITS.values())
    if not isinstance(rows, list) or len(rows) != expected_total:
        raise ValueError(f"authoritative manifest must contain exactly {expected_total:,} rows")
    counts = Counter(row.get("split") for row in rows)
    if dict(counts) != EXPECTED_SPLITS:
        raise ValueError(f"authoritative split counts differ: {dict(counts)}")
    ids = [row.get("area_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != expected_total:
        raise ValueError("authoritative manifest identities are missing or duplicated")
    return rows


def _resolve_shard(cache_manifest_path: Path, entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry.get("path", "")))
    return path if path.is_absolute() else cache_manifest_path.parent / path


def _validate_source_manifest(payload: Mapping[str, Any], row_count: int) -> None:
    expected = {
        "status": "complete", "sample_count": row_count,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "croma_checkpoint_sha256": CROMA_CHECKPOINT_SHA256,
        "croma_source_revision": CROMA_SOURCE_REVISION,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"source cache {field} mismatch")
    dimensions = payload.get("feature_dimensions") or {}
    if dimensions.get("physical") != 62 or dimensions.get("joint_croma") != 768 or dimensions.get("hybrid") != 830:
        raise ValueError("source cache feature dimensions mismatch")


def _validate_arrays(shard: Mapping[str, np.ndarray], expected_ids: list[str], expected_splits: list[str]) -> None:
    required = {"area_ids", "splits", "physical", "joint_croma"}
    if not required.issubset(shard):
        raise ValueError(f"source shard lacks arrays: {sorted(required.difference(shard))}")
    if shard["area_ids"].tolist() != expected_ids:
        raise ValueError("source shard image ordering mismatch")
    if shard["splits"].tolist() != expected_splits:
        raise ValueError("source shard split ordering mismatch")
    count = len(expected_ids)
    for name, shape in (("physical", (count, 62)), ("joint_croma", (count, 768))):
        values = shard[name]
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"source shard {name} shape, dtype, or finiteness mismatch")


def _validate_existing_hybrid(path: Path, *, checksum: str, ids: list[str], splits: list[str]) -> bool:
    if not path.is_file() or file_sha256(path) != checksum:
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != {"area_ids", "splits", "hybrid"}:
                return False
            return (shard["area_ids"].tolist() == ids and shard["splits"].tolist() == splits
                    and shard["hybrid"].shape == (len(ids), 830)
                    and shard["hybrid"].dtype == np.float32 and np.isfinite(shard["hybrid"]).all())
    except (OSError, ValueError):
        return False


def materialize_core(
    source_cache_manifest: Path,
    dataset_manifest: Path,
    output_root: Path,
    *,
    producer_version: str = "phase2d2_v1",
) -> dict[str, Any]:
    """Verify the existing cache and atomically create/reuse hybrid shards."""
    started = time.perf_counter()
    source_cache_manifest = Path(source_cache_manifest)
    dataset_manifest = Path(dataset_manifest)
    output_root = Path(output_root)
    rows = _load_authoritative_rows(dataset_manifest)
    source_payload = json.loads(source_cache_manifest.read_text(encoding="utf-8"))
    _validate_source_manifest(source_payload, len(rows))
    shard_entries = source_payload.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError("source cache has no shards")

    hybrid_root = output_root / "representations" / "hybrid"
    manifest_root = output_root / "receipts" / "shards"
    failures: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    cursor = 0
    reused = 0
    created = 0
    validation_seconds = 0.0
    hybrid_seconds = 0.0
    for ordinal, entry in enumerate(shard_entries):
        count = entry.get("rows")
        if type(count) is not int or count <= 0:
            raise ValueError("source shard row count is invalid")
        selected = rows[cursor:cursor + count]
        ids = [row["area_id"] for row in selected]
        splits = [row["split"] for row in selected]
        source_path = _resolve_shard(source_cache_manifest, entry)
        validation_started = time.perf_counter()
        if not source_path.is_file() or file_sha256(source_path) != entry.get("sha256"):
            raise ValueError(f"source shard checksum mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in ("area_ids", "splits", "physical", "joint_croma") if name in source.files}
            _validate_arrays(arrays, ids, splits)
            physical = np.ascontiguousarray(arrays["physical"], dtype=np.float32)
            joint = np.ascontiguousarray(arrays["joint_croma"], dtype=np.float32)
        validation_seconds += time.perf_counter() - validation_started

        shard_id = f"{cursor:05d}-{cursor + count - 1:05d}"
        hybrid_path = hybrid_root / f"{shard_id}.npz"
        shard_receipt_path = manifest_root / f"{shard_id}.json"
        prior = None
        if shard_receipt_path.is_file():
            try:
                prior = json.loads(shard_receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = None
        prior_checksum = (prior or {}).get("artifacts", {}).get("hybrid_830d", {}).get("artifact_sha256")
        if isinstance(prior_checksum, str) and _validate_existing_hybrid(
            hybrid_path, checksum=prior_checksum, ids=ids, splits=splits
        ):
            reused += 1
        else:
            if hybrid_path.exists() or shard_receipt_path.exists():
                raise ValueError(f"invalid existing shard requires explicit removal/regeneration: {shard_id}")
            hybrid_started = time.perf_counter()
            hybrid = np.ascontiguousarray(np.concatenate((physical, joint), axis=1), dtype=np.float32)
            if hybrid.shape != (count, 830) or not np.isfinite(hybrid).all():
                raise ValueError(f"invalid hybrid shard: {shard_id}")
            hybrid_root.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{shard_id}.", suffix=".npz.tmp", dir=hybrid_root)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    np.savez_compressed(stream, area_ids=np.asarray(ids), splits=np.asarray(splits), hybrid=hybrid)
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, hybrid_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            hybrid_seconds += time.perf_counter() - hybrid_started
            created += 1

        source_checksum = str(entry["sha256"])
        hybrid_checksum = file_sha256(hybrid_path)
        artifacts = {
            "physical_62d": _artifact(source_path, source_checksum, "physical", [count, 62], "pipeline3_physical_features"),
            "joint_croma_gap_768d": _artifact(source_path, source_checksum, "joint_croma", [count, 768], "official_croma"),
            "hybrid_830d": _artifact(hybrid_path, hybrid_checksum, "hybrid", [count, 830], "phase2d2_concat"),
        }
        shard_manifest = {
            "schema_version": RECEIPT_SCHEMA_VERSION, "shard_id": shard_id,
            "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
            "split": splits[0] if len(set(splits)) == 1 else "mixed", "sample_count": count,
            "start_index": cursor, "end_index": cursor + count - 1, "ordered_image_ids": ids,
            "ordered_splits": splits, "artifacts": artifacts, "producer": "representation_materialization.materialize_core",
            "producer_version": producer_version, "model": "official CROMA base",
            "model_version": CROMA_CHECKPOINT_SHA256,
            "preprocessing_version": source_payload.get("normalization_implementation"),
            "dataset_version": DATASET_FINGERPRINT,
        }
        _atomic_json(shard_receipt_path, shard_manifest)
        for index, (image_id, split) in enumerate(zip(ids, splits)):
            values_by_type = {
                "physical_62d": physical[index], "joint_croma_gap_768d": joint[index],
                "hybrid_830d": np.concatenate((physical[index], joint[index])).astype(np.float32, copy=False),
            }
            for representation_type, values in values_by_type.items():
                metadata = artifacts[representation_type]
                receipts.append({
                    "schema_version": RECEIPT_SCHEMA_VERSION, "image_id": image_id, "split": split,
                    "representation_type": representation_type, "variant": "default", "shard_id": shard_id,
                    "artifact_path": metadata["artifact_path"], "array_key": metadata["array_key"],
                    "sample_index": index, "shape": list(values.shape), "dtype": str(values.dtype),
                    "artifact_sha256": metadata["artifact_sha256"], "sample_sha256": sample_sha256(values),
                    "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
                    "producer": metadata["producer"], "model": "official CROMA base" if "croma" in representation_type else None,
                    "model_version": CROMA_CHECKPOINT_SHA256 if "croma" in representation_type or representation_type == "hybrid_830d" else None,
                    "preprocessing_version": source_payload.get("normalization_implementation"),
                    "provenance": {"source_cache_manifest_sha256": file_sha256(source_cache_manifest),
                                   "shard_checksum_is_not_sample_checksum": True},
                })
        groups.append(shard_manifest)
        cursor += count
    if cursor != len(rows):
        raise ValueError(f"source shards cover {cursor}, expected 5000")
    identities = [(item["image_id"], item["representation_type"], item["variant"]) for item in receipts]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate image/type/variant receipt identity")

    receipt_bytes = b"".join(canonical_bytes(item) + b"\n" for item in receipts)
    receipt_path = output_root / "receipts" / "records.jsonl"
    _atomic_bytes(receipt_path, receipt_bytes)
    result = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT, "sample_count": len(rows), "split_counts": EXPECTED_SPLITS,
        "core_representation_types": list(CORE_TYPES), "complete_core_count": len(rows),
        "source_cache_manifest_path": str(source_cache_manifest.resolve()),
        "source_cache_manifest_sha256": file_sha256(source_cache_manifest),
        "receipt_catalog_path": str(receipt_path.resolve()), "receipt_catalog_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_count": len(receipts), "shard_count": len(groups), "hybrid_shards_created": created,
        "hybrid_shards_reused": reused, "shards": groups, "failures": failures,
        "croma": {"model": "official CROMA base", "revision": CROMA_SOURCE_REVISION,
                  "checkpoint_sha256": CROMA_CHECKPOINT_SHA256,
                  "loader": "src.croma_adapter.CROMAAdapter", "inference_mode": True,
                  "configuration": {"modality": "both", "image_resolution": 120, "tokens": [225, 768]}},
        "physical": {"dimension": 62, "producer": "src.gee_features.LocalRasterFeatureProvider.extract",
                     "source_artifacts_reused": True},
        "hybrid": {"dimension": 830, "order": "physical_62d then joint_croma_gap_768d"},
        "timing_seconds": {"source_validation": validation_seconds, "hybrid_assembly": hybrid_seconds,
                           "total": time.perf_counter() - started},
        "storage_bytes": {"reused_source_shards": sum(Path(group["artifacts"]["physical_62d"]["artifact_path"]).stat().st_size for group in groups),
                          "new_hybrid_shards": sum(Path(group["artifacts"]["hybrid_830d"]["artifact_path"]).stat().st_size for group in groups)},
        "resumable": True, "atomic_writes": True, "sample_hash_algorithm": "sha256(dtype || canonical_shape || C_order_bytes)",
    }
    _atomic_json(output_root / "manifest.json", result)
    return result


def _artifact(path: Path, checksum: str, array_key: str, shape: list[int], producer: str) -> dict[str, Any]:
    return {"artifact_path": str(path.resolve()), "artifact_sha256": checksum, "artifact_size": path.stat().st_size,
            "array_key": array_key, "shape": shape, "dtype": "float32", "producer": producer}


def load_verified_receipts(materialization_manifest: Path) -> list[dict[str, Any]]:
    """Load and validate the trusted shard receipt catalog deterministically."""
    payload = json.loads(Path(materialization_manifest).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "complete":
        raise ValueError("materialization manifest is incomplete or unsupported")
    if payload.get("dataset_fingerprint") != DATASET_FINGERPRINT or payload.get("split_fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("materialization dataset identity mismatch")
    receipt_path = Path(payload["receipt_catalog_path"])
    content = receipt_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != payload.get("receipt_catalog_sha256"):
        raise ValueError("receipt catalog checksum mismatch")
    receipts = [json.loads(line) for line in content.splitlines() if line]
    if len(receipts) != payload.get("receipt_count"):
        raise ValueError("receipt catalog count mismatch")
    identities: set[tuple[str, str, str]] = set()
    verified_artifacts: dict[str, tuple[int, str]] = {}
    by_artifact: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for receipt in receipts:
        identity = (receipt.get("image_id"), receipt.get("representation_type"), receipt.get("variant"))
        if identity in identities:
            raise ValueError(f"duplicate receipt identity: {identity}")
        identities.add(identity)
        path = Path(receipt["artifact_path"])
        checksum = receipt.get("artifact_sha256")
        if checksum not in verified_artifacts:
            if not path.is_file() or file_sha256(path) != checksum:
                raise ValueError(f"receipt artifact checksum mismatch: {path}")
            verified_artifacts[checksum] = (path.stat().st_size, str(path))
        by_artifact.setdefault((str(path), receipt["array_key"]), []).append(receipt)
    for (path_text, key), artifact_receipts in by_artifact.items():
        with np.load(Path(path_text), allow_pickle=False) as shard:
            if key not in shard.files:
                raise ValueError("receipt array key mismatch")
            values = shard[key]
            for receipt in artifact_receipts:
                index = receipt["sample_index"]
                if type(index) is not int or index < 0 or index >= len(values):
                    raise ValueError("receipt sample index mismatch")
                sample = values[index]
                if list(sample.shape) != receipt["shape"] or str(sample.dtype) != receipt["dtype"]:
                    raise ValueError("receipt sample shape/dtype mismatch")
                if sample_sha256(sample) != receipt.get("sample_sha256"):
                    raise ValueError("receipt sample checksum mismatch")
    return receipts
