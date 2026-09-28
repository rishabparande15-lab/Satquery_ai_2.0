from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.representation_artifacts import file_sha256
import src.representation_materialization as materialization
from src.representation_catalog import build_catalog
from src.representation_linking import link_annotation
from src.receipt_catalog import validate_materialization_receipts


def _fixture(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(materialization, "EXPECTED_SPLITS", {"train": 2, "validation": 1, "test": 1})
    rows = [
        {"area_id": "a", "split": "train"}, {"area_id": "b", "split": "train"},
        {"area_id": "c", "split": "validation"}, {"area_id": "d", "split": "test"},
    ]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"dataset": "fixture", "dataset_fingerprint": materialization.DATASET_FINGERPRINT,
                                   "rows": rows}), encoding="utf-8")
    shard = tmp_path / "source.npz"
    physical = np.arange(4 * 62, dtype=np.float32).reshape(4, 62)
    joint = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
    np.savez_compressed(shard, area_ids=np.asarray([row["area_id"] for row in rows]),
                        splits=np.asarray([row["split"] for row in rows]), physical=physical,
                        joint_croma=joint)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "status": "complete", "sample_count": 4, "split_counts": materialization.EXPECTED_SPLITS,
        "dataset_fingerprint": materialization.DATASET_FINGERPRINT,
        "split_fingerprint": materialization.SPLIT_FINGERPRINT,
        "croma_checkpoint_sha256": materialization.CROMA_CHECKPOINT_SHA256,
        "croma_source_revision": materialization.CROMA_SOURCE_REVISION,
        "normalization_implementation": "fixture", "feature_dimensions": {"physical": 62, "joint_croma": 768, "hybrid": 830},
        "shards": [{"path": shard.name, "rows": 4, "sha256": file_sha256(shard)}],
    }), encoding="utf-8")
    return dataset, source, physical, joint


def test_core_shard_receipts_catalog_and_resume(tmp_path, monkeypatch):
    dataset, source, physical, joint = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    first = materialization.materialize_core(source, dataset, output)
    assert first["receipt_count"] == 12
    assert first["hybrid_shards_created"] == 1
    receipts = materialization.load_verified_receipts(output / "manifest.json")
    assert len(receipts) == 12
    assert len(validate_materialization_receipts(output / "manifest.json")) == 12
    with np.load(output / "representations/hybrid/00000-00003.npz", allow_pickle=False) as shard:
        np.testing.assert_array_equal(shard["hybrid"], np.concatenate((physical, joint), axis=1))
    second = materialization.materialize_core(source, dataset, output)
    assert second["hybrid_shards_reused"] == 1
    catalog = build_catalog(dataset, materialization_manifest_paths=(output / "manifest.json",),
                            representation_types=materialization.CORE_TYPES)
    assert catalog["report"]["counts_by_status"] == {"VERIFIED": 12}
    assert catalog["report"]["complete_core_image_count"] == 4
    row = catalog["records"][0]
    linked = link_annotation(
        {"annotation_id": "ann", "image_id": row.image_id, "split": row.split,
         "annotation_type": "binary_qa", "task_type": "binary_qa", "provenance": {}},
        {row.image_id: {"split": row.split}},
        [{"reference_id": row.representation_id, "representation_type": row.representation_type,
          "scene_id": row.image_id, "split": row.split, "producer": row.producer,
          "producer_version": row.model_version, "preprocessing_version": row.preprocessing_version,
          "checksum_sha256": row.checksum, "artifact_uri": row.logical_uri,
          "spatial_reference": row.spatial_semantics}],
    )
    assert linked.validation_status == "valid"


def test_checksum_order_split_and_interrupted_shards_fail_closed(tmp_path, monkeypatch):
    dataset, source, _, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(source.read_text())
    payload["shards"][0]["sha256"] = "0" * 64
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        materialization.materialize_core(source, dataset, tmp_path / "bad")

    dataset, source, _, _ = _fixture(tmp_path / "again", monkeypatch)
    output = tmp_path / "interrupted"
    hybrid = output / "representations/hybrid/00000-00003.npz"
    hybrid.parent.mkdir(parents=True)
    hybrid.write_bytes(b"partial")
    with pytest.raises(ValueError, match="explicit removal/regeneration"):
        materialization.materialize_core(source, dataset, output)


def test_receipt_tampering_and_duplicate_identity_are_rejected(tmp_path, monkeypatch):
    dataset, source, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    result = materialization.materialize_core(source, dataset, output)
    receipt_path = output / "receipts/records.jsonl"
    lines = receipt_path.read_bytes().splitlines()
    receipt_path.write_bytes(b"\n".join(lines + [lines[0]]) + b"\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    content = receipt_path.read_bytes()
    manifest["receipt_catalog_sha256"] = hashlib.sha256(content).hexdigest()
    manifest["receipt_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate receipt identity"):
        materialization.load_verified_receipts(manifest_path)
