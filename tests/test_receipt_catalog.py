from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import time

import numpy as np
import pytest

from src.architecture_contracts import ArtifactRef, RepresentationSet, SceneBundle
from src.live_representation_persistence import (
    PersistableRepresentation,
    RepresentationPersistencePolicy,
    RepresentationPersistenceSession,
)
from src.receipt_catalog import LifecycleStatus, ReceiptCatalog
from src.representation_artifacts import ArtifactResolver, artifact_ref_from_file


def _persist(root, *, scene="scene-a", run="run-1", selected=PersistableRepresentation.OPTICAL_CROMA_TOKENS, value=None):
    policy = RepresentationPersistencePolicy(
        enabled=True,
        requested=frozenset({selected}),
        storage_root=root,
        namespace="live-test",
    )
    session = RepresentationPersistenceSession(policy)
    if value is None:
        value = np.arange(225 * 768, dtype=np.float32).reshape(225, 768)
    session.capture(selected, value, producer_version="checkpoint", preprocessing_version="normalization")
    return session.finalize(scene_id=scene, run_id=run, crs="EPSG:32633", provenance={})


def _catalog(root):
    return ReceiptCatalog(root / "receipt_catalog.json", ArtifactResolver({"live-test": root}))


def test_persistence_creates_catalog_and_registers_verified_reference(tmp_path):
    outcome = _persist(tmp_path)
    catalog = _catalog(tmp_path)
    records = catalog.lookup(scene_id="scene-a", run_id="run-1")
    assert len(records) == 1
    assert records[0].representation_ref == outcome.references[0]
    assert records[0].artifact_ref.checksum_sha256 == outcome.references[0].artifact.checksum_sha256
    assert records[0].status is LifecycleStatus.ACTIVE
    assert json.loads((tmp_path / "receipt_catalog.json").read_text())["catalog_version"] == 1


def test_duplicate_registration_is_idempotent_and_conflicting_identity_is_rejected(tmp_path):
    outcome = _persist(tmp_path)
    catalog = _catalog(tmp_path)
    first = catalog.register(outcome.references[0], outcome.receipt)
    assert catalog.register(outcome.references[0], outcome.receipt) == first
    assert len(catalog.lookup()) == 1

    other_path = tmp_path / "scene-a" / "run-1" / "other.npy"
    np.save(other_path, np.ones((225, 768), dtype=np.float32), allow_pickle=False)
    other_artifact = artifact_ref_from_file(
        other_path,
        artifact_id=first.artifact_id,
        artifact_type=first.artifact_ref.artifact_type,
        logical_uri="artifact://live-test/scene-a/run-1/other.npy",
        producer="CROMAAdapter",
        provenance_reference=outcome.receipt.uri,
        format="npy",
    )
    other_artifact = replace(other_artifact, run_reference="run-1")
    conflict = replace(first.representation_ref, artifact=other_artifact)
    with pytest.raises(ValueError, match="conflicting catalog registration"):
        catalog.register(conflict, outcome.receipt)


def test_lookup_filters_scene_representation_modality_run_id_and_logical_identity(tmp_path):
    first = _persist(tmp_path, scene="scene-a", run="run-1")
    second = _persist(
        tmp_path,
        scene="scene-b",
        run="run-2",
        selected=PersistableRepresentation.SAR_CROMA_GAP,
        value=np.ones(768, dtype=np.float32),
    )
    catalog = _catalog(tmp_path)
    assert catalog.lookup(scene_id="scene-a")[-1].representation_ref == first.references[0]
    assert catalog.lookup(scene_id="scene-b", representation_type="sar_croma", modality="sar", run_id="run-2")[-1].representation_ref == second.references[0]
    assert catalog.lookup(scene_id="scene-a", run_id="run-2") == ()
    assert catalog.lookup(artifact_id=first.references[0].artifact.artifact_id)[0].scene_id == "scene-a"
    assert catalog.lookup(artifact_uri=second.references[0].artifact.uri)[0].scene_id == "scene-b"


def test_validate_and_materialize_use_existing_resolver(tmp_path):
    outcome = _persist(tmp_path)
    catalog = _catalog(tmp_path)
    record = catalog.lookup()[0]
    assert catalog.validate(record).valid is True
    material = catalog.materialize(record.artifact_id)
    assert material.shape == (225, 768)
    assert material.dtype == np.float32
    np.testing.assert_array_equal(material, np.arange(225 * 768, dtype=np.float32).reshape(225, 768))
    assert outcome.references[0].spatial_reference.token_grid == (15, 15)


@pytest.mark.parametrize("mutation", ["tamper", "truncate", "delete"])
def test_corruption_is_persistently_invalidated_and_never_returned_as_active(tmp_path, mutation):
    outcome = _persist(tmp_path)
    resolver = ArtifactResolver({"live-test": tmp_path})
    path = resolver.resolve(outcome.references[0].artifact)
    if mutation == "tamper":
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
    elif mutation == "truncate":
        path.write_bytes(path.read_bytes()[:64])
    else:
        path.unlink()
    catalog = _catalog(tmp_path)
    result = catalog.validate(outcome.references[0].artifact.artifact_id)
    assert result.valid is False
    assert result.record.status is LifecycleStatus.INVALID
    assert catalog.lookup() == ()
    assert catalog.lookup(active_only=False)[0].invalid_reason
    reopened = _catalog(tmp_path)
    assert reopened.lookup() == ()
    with pytest.raises(ValueError, match="not reusable"):
        reopened.materialize(result.record.artifact_id)


def test_expiration_changes_lifecycle_without_deleting_artifact(tmp_path):
    outcome = _persist(tmp_path)
    catalog = _catalog(tmp_path)
    catalog_path = tmp_path / "second_catalog.json"
    expiring = ReceiptCatalog(catalog_path, ArtifactResolver({"live-test": tmp_path}))
    deadline = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    record = expiring.register(outcome.references[0], outcome.receipt, expires_at=deadline)
    assert expiring.lookup() == ()
    expired = expiring.lookup(active_only=False)[0]
    assert expired.status is LifecycleStatus.EXPIRED
    assert ArtifactResolver({"live-test": tmp_path}).resolve(record.artifact_ref).is_file()


def test_restart_reopen_discovers_validates_and_materializes_prior_artifact(tmp_path):
    outcome = _persist(tmp_path)
    del outcome
    reopened = _catalog(tmp_path)
    found = reopened.lookup(scene_id="scene-a", representation_type="optical_croma", spatial_level="token")
    assert len(found) == 1
    assert reopened.validate(found[0]).valid
    assert reopened.materialize(found[0]).shape == (225, 768)


def test_scene_bundle_discovery_preserves_representation_set_responsibility(tmp_path):
    _persist(tmp_path, scene="scene-a", run="run-1")
    catalog = _catalog(tmp_path)
    scene = SceneBundle(scene_id="scene-a", analysis_id="run-1", available_modalities=("optical",), representations=RepresentationSet())
    discovered = catalog.attach_to_scene(scene)
    assert scene.representations.references == ()
    assert len(discovered.representations.find("optical_croma", spatial_level="token", addressable_only=True)) == 1
    assert catalog.attach_to_scene(SceneBundle(scene_id="scene-b")).representations.references == ()


def test_t1_t2_runs_coexist_without_cross_run_collision(tmp_path):
    _persist(tmp_path, scene="temporal-scene", run="t1", value=np.zeros((225, 768), dtype=np.float32))
    _persist(tmp_path, scene="temporal-scene", run="t2", value=np.ones((225, 768), dtype=np.float32))
    catalog = _catalog(tmp_path)
    t1 = catalog.lookup(scene_id="temporal-scene", run_id="t1")
    t2 = catalog.lookup(scene_id="temporal-scene", run_id="t2")
    assert len(t1) == len(t2) == 1
    assert t1[0].artifact_ref.uri != t2[0].artifact_ref.uri
    assert not np.array_equal(catalog.materialize(t1[0]), catalog.materialize(t2[0]))


def test_registration_rejects_untrusted_or_incomplete_inputs(tmp_path):
    outcome = _persist(tmp_path)
    catalog = _catalog(tmp_path)
    with pytest.raises(TypeError):
        catalog.register(r"C:\untrusted\array.npy", outcome.receipt)
    incomplete_artifact = replace(outcome.references[0].artifact, size_bytes=None)
    incomplete = replace(outcome.references[0], artifact=incomplete_artifact)
    with pytest.raises(ValueError, match="checksum and size"):
        ReceiptCatalog(tmp_path / "other.json", catalog.resolver).register(incomplete, outcome.receipt)
    unknown = replace(
        outcome.references[0].artifact,
        uri="artifact://uncontrolled/scene-a/run-1/croma.npy",
    )
    unknown_ref = replace(outcome.references[0], artifact=unknown)
    with pytest.raises(LookupError, match="unconfigured"):
        ReceiptCatalog(tmp_path / "other.json", catalog.resolver).register(unknown_ref, outcome.receipt)


def test_registration_rejects_valid_but_inconsistent_receipt_metadata(tmp_path):
    outcome = _persist(tmp_path)
    receipt_path = tmp_path / "scene-a" / "run-1" / "unrelated-receipt.json"
    receipt_path.write_text(json.dumps({
        "status": "complete", "scene_id": "other-scene", "run_id": "run-1", "representations": []
    }), encoding="utf-8")
    unrelated = artifact_ref_from_file(
        receipt_path,
        artifact_id="unrelated-receipt",
        artifact_type=outcome.receipt.artifact_type,
        logical_uri="artifact://live-test/scene-a/run-1/unrelated-receipt.json",
        producer="test",
        format="json",
    )
    reference = replace(
        outcome.references[0],
        provenance_reference=unrelated.uri,
        artifact=replace(outcome.references[0].artifact, provenance_reference=unrelated.uri),
    )
    with pytest.raises(ValueError, match="conflicts with receipt scene/run"):
        ReceiptCatalog(tmp_path / "other.json", ArtifactResolver({"live-test": tmp_path})).register(reference, unrelated)


@pytest.mark.parametrize("payload", [
    "not-json",
    json.dumps({"catalog_version": 1, "records": [{}]}),
    json.dumps({"catalog_version": 999, "records": []}),
])
def test_malformed_catalog_records_fail_closed(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        ReceiptCatalog(path, ArtifactResolver({"live-test": tmp_path}))


def test_small_local_lookup_cost_is_practical(tmp_path):
    _persist(tmp_path)
    catalog = _catalog(tmp_path)
    started = time.perf_counter()
    for _ in range(1000):
        assert catalog.lookup(scene_id="scene-a", run_id="run-1")
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0
