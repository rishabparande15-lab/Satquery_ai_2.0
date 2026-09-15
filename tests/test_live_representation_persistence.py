from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest
import torch

from src.architecture_contracts import adapt_run_analysis_output
from src.analysis_engine import run_analysis
from src.deterministic_scene_analysis import Pipeline3AnalysisAdapter
from src.live_representation_persistence import (
    CROMA_REPRESENTATIONS,
    PersistableRepresentation,
    RepresentationPersistencePolicy,
    RepresentationPersistenceSession,
)
from src.representation_artifacts import ArtifactResolver


def _policy(tmp_path, *selected, max_total_bytes=64 * 1024 * 1024):
    return RepresentationPersistencePolicy(
        enabled=True,
        requested=frozenset(selected),
        storage_root=tmp_path,
        namespace="live-test",
        max_total_bytes=max_total_bytes,
    )


def _finalize(session, *, run_id="run-1"):
    return session.finalize(
        scene_id="scene-1",
        run_id=run_id,
        crs="EPSG:32633",
        provenance={
            "analysis_id": run_id,
            "input": {"sample_id": "scene-1"},
            "preprocessing": ["existing"],
            "representation": {"croma_source": "official pretrained CROMA base"},
        },
    )


def test_policy_is_disabled_by_default_and_cannot_write(tmp_path):
    policy = RepresentationPersistencePolicy(
        requested=frozenset({PersistableRepresentation.PHYSICAL_FEATURES}),
        storage_root=tmp_path,
        namespace="disabled",
    )
    session = RepresentationPersistenceSession(policy)
    session.capture(PersistableRepresentation.PHYSICAL_FEATURES, np.ones(2, dtype=np.float32))
    assert policy.enabled is False
    assert session.captured(PersistableRepresentation.PHYSICAL_FEATURES) is None
    assert _finalize(session) is None
    assert list(tmp_path.iterdir()) == []


def test_run_analysis_default_and_disabled_session_keep_the_legacy_schema(tmp_path):
    request = {"query": "optical", "files": {}}
    direct = run_analysis(request)
    policy = RepresentationPersistencePolicy(
        requested=frozenset({PersistableRepresentation.PHYSICAL_FEATURES}),
        storage_root=tmp_path,
        namespace="disabled",
    )
    disabled = run_analysis(request, persistence_session=RepresentationPersistenceSession(policy))
    assert direct.keys() == disabled.keys()
    assert "representation_persistence" not in direct
    assert "representation_persistence" not in disabled
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("changes", [
    {"enabled": True, "requested": frozenset(), "storage_root": "root", "namespace": "test"},
    {"enabled": True, "requested": {PersistableRepresentation.PHYSICAL_FEATURES}, "namespace": "test"},
    {"enabled": True, "requested": {PersistableRepresentation.PHYSICAL_FEATURES}, "storage_root": "root", "namespace": "../bad"},
    {"enabled": True, "requested": {PersistableRepresentation.PHYSICAL_FEATURES}, "storage_root": "root", "namespace": "test", "checksum_required": False},
    {"max_total_bytes": 0},
])
def test_policy_requires_explicit_safe_bounded_enablement(changes):
    with pytest.raises(ValueError):
        RepresentationPersistencePolicy(**changes)


def test_selected_filter_captures_only_explicit_representation(tmp_path):
    session = RepresentationPersistenceSession(_policy(tmp_path, PersistableRepresentation.PHYSICAL_FEATURES))
    physical = np.arange(4, dtype=np.float32)
    session.capture(PersistableRepresentation.PHYSICAL_FEATURES, physical)
    session.capture(PersistableRepresentation.RAW_OPTICAL, np.zeros((12, 2, 2), dtype=np.float32))
    assert np.array_equal(session.captured(PersistableRepresentation.PHYSICAL_FEATURES), physical)
    assert session.captured(PersistableRepresentation.RAW_OPTICAL) is None


def test_persistence_generates_receipt_refs_and_exactly_materializes_content(tmp_path):
    selected = PersistableRepresentation.OPTICAL_CROMA_TOKENS
    source = np.arange(225 * 768, dtype=np.float32).reshape(225, 768)
    session = RepresentationPersistenceSession(_policy(tmp_path, selected))
    session.capture(
        selected,
        source,
        producer_version="checkpoint-sha256",
        preprocessing_version="croma-normalization-v1",
    )
    outcome = _finalize(session)
    assert outcome is not None
    assert outcome.writes == 2
    assert outcome.reused == 0
    assert len(outcome.references) == 1
    reference = outcome.references[0]
    assert reference.status == "available"
    assert reference.addressable
    assert reference.shape == (225, 768)
    assert reference.dtype == "float32"
    assert reference.spatial_reference.token_grid == (15, 15)
    assert reference.modality == "optical"
    assert reference.producer == "CROMAAdapter"
    assert reference.provenance_reference == outcome.receipt.uri
    resolver = ArtifactResolver({"live-test": tmp_path})
    loaded = resolver.load_numpy(reference)
    assert np.array_equal(loaded, source)
    assert np.array_equal(loaded, session.captured(selected))
    receipt_path = resolver.resolve(outcome.receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["run_id"] == "run-1"
    assert receipt["representations"][0]["reference"]["artifact"]["checksum_sha256"] == reference.artifact.checksum_sha256
    assert str(tmp_path) not in json.dumps(receipt)


def test_croma_capture_preserves_six_modality_outputs_without_batch_dimension(tmp_path):
    session = RepresentationPersistenceSession(_policy(tmp_path, *CROMA_REPRESENTATIONS))
    outputs = {
        "optical_encodings": torch.zeros((1, 225, 768)),
        "SAR_encodings": torch.ones((1, 225, 768)),
        "joint_encodings": torch.full((1, 225, 768), 2.0),
        "optical_GAP": torch.zeros((1, 768)),
        "SAR_GAP": torch.ones((1, 768)),
        "joint_GAP": torch.full((1, 768), 2.0),
    }
    session.capture_croma(outputs, producer_version="checkpoint", preprocessing_version="normalization")
    outcome = _finalize(session)
    token_shapes = sorted(reference.shape for reference in outcome.references if reference.spatial_reference.level == "token")
    gap_shapes = sorted(reference.shape for reference in outcome.references if reference.spatial_reference.level == "scene")
    assert token_shapes == [(225, 768)] * 3
    assert gap_shapes == [(768,)] * 3
    assert outcome.writes == 7


def test_identical_artifacts_are_reused_and_conflicting_writes_are_rejected(tmp_path):
    selected = PersistableRepresentation.PHYSICAL_FEATURES
    first = RepresentationPersistenceSession(_policy(tmp_path, selected))
    first.capture(selected, np.array([1, 2], dtype=np.float32))
    assert _finalize(first).writes == 2
    same = RepresentationPersistenceSession(_policy(tmp_path, selected))
    same.capture(selected, np.array([1, 2], dtype=np.float32))
    outcome = _finalize(same)
    assert outcome.writes == 0
    assert outcome.reused == 2
    conflict = RepresentationPersistenceSession(_policy(tmp_path, selected))
    conflict.capture(selected, np.array([3, 4], dtype=np.float32))
    with pytest.raises(FileExistsError, match="immutable artifact conflict"):
        _finalize(conflict)


def test_external_tampering_is_detected_by_checksum(tmp_path):
    selected = PersistableRepresentation.PHYSICAL_FEATURES
    session = RepresentationPersistenceSession(_policy(tmp_path, selected))
    session.capture(selected, np.array([1, 2], dtype=np.float32))
    outcome = _finalize(session)
    resolver = ArtifactResolver({"live-test": tmp_path})
    path = resolver.resolve(outcome.references[0].artifact)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        resolver.load_numpy(outcome.references[0])


def test_storage_limit_and_logical_identity_fail_closed_before_artifact_write(tmp_path):
    selected = PersistableRepresentation.PHYSICAL_FEATURES
    session = RepresentationPersistenceSession(_policy(tmp_path, selected, max_total_bytes=1))
    session.capture(selected, np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="max_total_bytes"):
        _finalize(session)
    assert not (tmp_path / "scene-1" / "run-1").exists()
    safe = RepresentationPersistenceSession(_policy(tmp_path, selected))
    safe.capture(selected, np.ones(1, dtype=np.float32))
    with pytest.raises(ValueError, match="safe logical identifiers"):
        safe.finalize(scene_id="../escape", run_id="run", crs=None, provenance={})


def test_object_arrays_and_duplicate_capture_are_rejected(tmp_path):
    selected = PersistableRepresentation.PHYSICAL_FEATURES
    session = RepresentationPersistenceSession(_policy(tmp_path, selected))
    with pytest.raises(ValueError, match="object arrays"):
        session.capture(selected, np.array([object()], dtype=object))
    session.capture(selected, np.ones(1, dtype=np.float32))
    with pytest.raises(ValueError, match="captured twice"):
        session.capture(selected, np.ones(1, dtype=np.float32))


def _legacy_result():
    return {
        "schema_version": "2.0", "analysis_id": "run-1", "timestamp": "now", "status": "completed",
        "query_interpretation": {"task": "joint_optical_sar_analysis", "modalities": ["optical", "sar"]},
        "data_retrieval": {"source": "local", "assets": {"sample_id": "scene-1"}},
        "validation": {"checks": {"optical_sar_compatibility": {"crs": True}}},
        "data_cube": {"modalities": ["optical", "sar"], "spatial_dimensions": [120, 120], "metadata": {}},
        "features": {
            "spectral": {"status": "computed", "dimension": 62},
            "deep": {"status": "computed", "source": "CROMA", "pooled_dimension": 2304,
                     "representations": {"optical_encodings": [1, 225, 768], "SAR_encodings": [1, 225, 768]}},
            "hybrid": {"status": "computed", "dimension": 192},
        },
        "spatial_evidence": {"status": "AVAILABLE", "schema_version": "spatial_evidence_v1",
                             "scene": {"sample_id": "scene-1", "crs": "EPSG:32633", "image_shape": [120, 120]},
                             "regions": [], "claims": [], "provenance": {}},
        "interpretation": {"status": "ANSWERED"}, "model_results": {"prediction": None},
        "evidence": [], "confidence": {"model_confidence": None}, "preprocessing": [],
        "execution_trace": [], "warnings": [], "error": None, "temporal": {"status": "not requested"},
    }


def test_scene_bundle_replaces_only_successfully_persisted_reference(tmp_path):
    selected = PersistableRepresentation.OPTICAL_CROMA_TOKENS
    session = RepresentationPersistenceSession(_policy(tmp_path, selected))
    session.capture(selected, np.zeros((225, 768), dtype=np.float32))
    _finalize(session)
    adapted = adapt_run_analysis_output(_legacy_result(), {"sample_id": "scene-1"})
    updated = Pipeline3AnalysisAdapter._attach_persisted_references(adapted, session)
    optical = updated.scene.representations.find("optical_croma", spatial_level="token")
    sar = updated.scene.representations.find("sar_croma", spatial_level="token")
    assert optical[0].status == "available"
    assert sar[0].status == "not_addressable"
    assert updated.scene.provenance["representation_persistence"]["receipt"]["uri"] == session.outcome.receipt.uri
    assert any(item.get("artifact_type") == "provenance_receipt" for item in updated.task_result.artifacts)


def test_policy_and_outcome_contracts_are_immutable(tmp_path):
    policy = _policy(tmp_path, PersistableRepresentation.PHYSICAL_FEATURES)
    with pytest.raises(FrozenInstanceError):
        policy.enabled = False
    session = RepresentationPersistenceSession(policy)
    session.capture(PersistableRepresentation.PHYSICAL_FEATURES, np.ones(1, dtype=np.float32))
    outcome = _finalize(session)
    with pytest.raises(FrozenInstanceError):
        outcome.total_bytes = 0
