from __future__ import annotations

import json

import numpy as np
import pytest

from src.representation_artifacts import ArtifactResolver
from src.representation_catalog import CatalogRecord, build_catalog, records_from_receipt, write_catalog
from src.representation_catalog import _modality_for


def _manifest(path):
    path.write_text(json.dumps({"dataset": "fixture", "rows": [
        {"area_id": "area-a", "split": "train"},
        {"area_id": "area-b", "split": "test"},
    ]}), encoding="utf-8")


def _receipt(root, *, sample="area-a", split="train", tamper=False):
    array = root / "values.npy"
    np.save(array, np.ones((2, 3), dtype=np.float32), allow_pickle=False)
    from src.representation_artifacts import file_sha256
    checksum = file_sha256(array)
    if tamper:
        array.write_bytes(array.read_bytes() + b"tampered")
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({"status": "complete", "sample_id": sample, "split": split,
                                   "arrays": {"physical_features": {"file": array.name, "shape": [2, 3], "dtype": "float32", "sha256": checksum}}}), encoding="utf-8")
    return receipt


def test_catalog_emits_missing_rows_for_every_manifest_image_and_type(tmp_path):
    manifest = tmp_path / "dataset.json"
    _manifest(manifest)
    catalog = build_catalog(manifest, representation_types=("physical_features", "hybrid"))
    assert len(catalog["records"]) == 4
    assert catalog["report"]["manifest_image_count"] == 2
    assert catalog["report"]["represented_image_count"] == 0
    assert catalog["report"]["counts_by_status"] == {"MISSING": 4}


def test_verified_receipt_requires_manifest_identity_and_checksum(tmp_path):
    manifest = tmp_path / "dataset.json"
    _manifest(manifest)
    receipt = _receipt(tmp_path)
    records = records_from_receipt(receipt, manifest={"area-a": {"split": "train", "dataset_version": "fixture"}},
                                    resolver=ArtifactResolver({"test": tmp_path}))
    assert records[0].availability == "VERIFIED"
    assert records[0].checksum and records[0].shape == (2, 3)
    outside = records_from_receipt(_receipt(tmp_path, sample="outside"), manifest={"area-a": {"split": "train"}})
    assert outside[0].availability == "PROVENANCE_MISMATCH"


def test_checksum_and_split_mismatch_are_explicit(tmp_path):
    manifest = tmp_path / "dataset.json"
    _manifest(manifest)
    tampered = records_from_receipt(_receipt(tmp_path, tamper=True), manifest={"area-a": {"split": "train"}})
    assert tampered[0].availability == "INVALID"
    mismatch = records_from_receipt(_receipt(tmp_path, split="test"), manifest={"area-a": {"split": "train"}})
    assert mismatch[0].availability == "SPLIT_MISMATCH"


def test_catalog_serialization_is_deterministic(tmp_path):
    manifest = tmp_path / "dataset.json"
    _manifest(manifest)
    first = build_catalog(manifest, representation_types=("hybrid", "physical_features"))
    second = build_catalog(manifest, representation_types=("physical_features", "hybrid"))
    assert first["canonical_bytes"] == second["canonical_bytes"]
    write_catalog(first, tmp_path / "out")
    assert json.loads((tmp_path / "out/manifest.json").read_text())["records_sha256"] == first["report"]["catalog_sha256"]


def test_catalog_record_rejects_invalid_status_and_shape():
    with pytest.raises(ValueError):
        CatalogRecord("area", "train", "hybrid", None, None, None, None, None, (0,), None, None, None, None, None, None, None, None, {}, "MISSING", "BOGUS")


def test_shard_backed_records_verify_identity_split_checksum_and_slice(tmp_path):
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({"dataset": "fixture", "dataset_fingerprint": "fp", "rows": [
        {"area_id": "area-a", "split": "train"}, {"area_id": "area-b", "split": "test"}]}), encoding="utf-8")
    shard = tmp_path / "00000-00001.npz"
    np.savez_compressed(shard,
                        area_ids=np.asarray(["area-a", "area-b"]),
                        splits=np.asarray(["train", "test"]),
                        physical=np.ones((2, 62), dtype=np.float32),
                        optical_croma=np.ones((2, 768), dtype=np.float32),
                        sar_croma=np.ones((2, 768), dtype=np.float32),
                        joint_croma=np.ones((2, 768), dtype=np.float32))
    from src.representation_artifacts import file_sha256
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"status": "complete", "sample_count": 2,
                                 "dataset_fingerprint": "fp", "shards": [
                                     {"path": shard.name, "rows": 2, "sha256": file_sha256(shard)}]}), encoding="utf-8")
    catalog = build_catalog(manifest, shard_manifest_paths=(cache,), representation_types=("physical_features",))
    assert catalog["report"]["shard_images_verified"] == 2
    assert catalog["report"]["counts_by_status"] == {"VERIFIED": 2}
    assert catalog["records"][0].shard_ref["sample_index"] == 0


def test_shard_split_mismatch_fails_closed(tmp_path):
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({"dataset": "fixture", "dataset_fingerprint": "fp", "rows": [
        {"area_id": "area-a", "split": "train"}]}), encoding="utf-8")
    shard = tmp_path / "shard.npz"
    np.savez_compressed(shard, area_ids=np.asarray(["area-a"]), splits=np.asarray(["test"]),
                        physical=np.ones((1, 62), dtype=np.float32), optical_croma=np.ones((1, 768), dtype=np.float32),
                        sar_croma=np.ones((1, 768), dtype=np.float32), joint_croma=np.ones((1, 768), dtype=np.float32))
    from src.representation_artifacts import file_sha256
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"status": "complete", "sample_count": 1, "dataset_fingerprint": "fp",
                                 "shards": [{"path": shard.name, "rows": 1, "sha256": file_sha256(shard)}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="split mismatch"):
        build_catalog(manifest, shard_manifest_paths=(cache,), representation_types=("physical_features",))


def test_canonical_core_types_retain_joint_modality():
    assert _modality_for("physical_62d") == "optical_sar"
    assert _modality_for("joint_croma_gap_768d") == "optical_sar"
    assert _modality_for("hybrid_830d") == "optical_sar"
