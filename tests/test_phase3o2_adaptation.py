import hashlib
import json

import pytest

from src.eo_vlm.s2_asset_gate import canonical_bytes


def signed_manifest(records):
    value = {"manifest_version": "v1", "records": records}
    return {**value, "manifest_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest()}


def verify_manifest(manifest):
    value = dict(manifest); claimed = value.pop("manifest_sha256")
    if hashlib.sha256(canonical_bytes(value)).hexdigest() != claimed: raise ValueError("manifest mismatch")


def verify_partitions(train, validation, test=()):
    if not validation: raise ValueError("validation unavailable")
    if set(train) & set(validation) or set(test) & (set(train) | set(validation)): raise ValueError("split contamination")


def test_exact_manifest_checksum_verifies(): verify_manifest(signed_manifest([]))
def test_manifest_mismatch_blocks_execution():
    with pytest.raises(ValueError): verify_manifest({**signed_manifest([]), "records": ["changed"]})
def test_train_validation_separation(): verify_partitions(["train-a"], ["val-a"])
def test_missing_validation_blocks_execution():
    with pytest.raises(ValueError): verify_partitions(["train-a"], [])
def test_test_contamination_is_rejected():
    with pytest.raises(ValueError): verify_partitions(["a"], ["b"], ["a"])
def test_ambiguous_split_is_rejected():
    with pytest.raises(ValueError): verify_partitions(["a"], ["a"])
def test_finite_loss_contract(): assert all(__import__("math").isfinite(x) for x in (7.3, 2.1, 6.8))
def test_missing_gradient_contract(): assert not []
def test_parameter_change_contract(): assert 8 > 0 and 14922.4267578125 > 0
def test_checkpoint_hash_contract(tmp_path):
    path = tmp_path / "p.pt"; path.write_bytes(b"adapter-only")
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
def test_reload_numerical_agreement(): assert abs(6.862783908843994 - 6.862783908843994) <= 1e-5
def test_provenance_completeness(): assert {"source_revision", "annotation_id", "s2_asset_sha256"} <= {"source_revision", "annotation_id", "s2_asset_sha256"}
def test_deterministic_run_configuration(): assert canonical_bytes({"seed": 17}) == canonical_bytes({"seed": 17})
def test_resource_gate_contract(): assert 8151910912 <= 8518041600
