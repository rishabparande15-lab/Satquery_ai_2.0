import pytest

from src.architecture_contracts import RepresentationRef, RepresentationType
from src.eo_vlm.temporal_contracts import (
    NOT_COMPUTED, UNKNOWN, ContractOnlyTemporalFusionAdapter, SpatialCorrespondence,
    SpatialCorrespondenceStatus, TemporalEvidenceItem, TemporalEvidenceKind,
    TemporalFrame, TemporalFusionUnavailable, TemporalInput, TemporalMetadata,
    TemporalModality, TemporalOrderStatus, TemporalProvenance, TemporalRepresentation,
    TemporalResourceEstimate, TemporalSequenceInput,
)


def provenance(name="source"):
    return TemporalProvenance(name, source_revision="pinned-revision", attributes={"split": "structural"})


def frame(identity, modality=TemporalModality.S2, shape=(12, 120, 120)):
    return TemporalFrame(identity, modality, provenance(identity), tensor_shape=shape)


def temporal_input(**overrides):
    values = {
        "t1": frame("t1"), "t2": frame("t2"),
        "temporal_order": TemporalOrderStatus.VERIFIED,
        "spatial_status": SpatialCorrespondenceStatus.VERIFIED,
        "spatial_correspondence": SpatialCorrespondence.SPATIALLY_CORRESPONDING,
        "temporal_metadata": TemporalMetadata(), "provenance": provenance("pair"),
    }
    values.update(overrides)
    return TemporalInput(**values)


def representation_ref(identity):
    return RepresentationRef(identity, identity, RepresentationType.RAW_OPTICAL, status="unavailable")


def test_s2_pair_contract_preserves_order_identity_and_deterministic_fingerprint():
    value = temporal_input()
    assert value.modalities == ("s2", "s2")
    assert value.fingerprint() == temporal_input().fingerprint()
    assert value.to_dict()["temporal_order"] == "TEMPORAL_ORDER_VERIFIED"


def test_sar_and_cross_modality_contracts_are_interface_only():
    sar = temporal_input(t1=frame("s1a", TemporalModality.S1, (2, 120, 120)), t2=frame("s1b", TemporalModality.S1, (2, 120, 120)))
    cross = temporal_input(t1=frame("optical", TemporalModality.OPTICAL, (3, 256, 256)), t2=frame("sar", TemporalModality.SAR, (2, 256, 256)))
    assert sar.modalities == ("s1", "s1")
    assert cross.modalities == ("optical", "sar")


def test_frame_shape_and_identity_validation_fail_closed():
    with pytest.raises(ValueError, match="s2 tensor_shape"):
        frame("bad", TemporalModality.S2, (3, 120, 120))
    with pytest.raises(ValueError, match="distinct"):
        temporal_input(t2=frame("t1"))


def test_temporal_metadata_and_spatial_status_validation():
    with pytest.raises(ValueError, match="ISO-8601"):
        TemporalMetadata(t1_timestamp="yesterday", t2_timestamp="2026-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="unverified"):
        temporal_input(spatial_status=SpatialCorrespondenceStatus.UNKNOWN)
    with pytest.raises(ValueError, match="verified"):
        temporal_input(spatial_correspondence=SpatialCorrespondence.UNKNOWN)


def test_representation_keeps_features_not_computed_and_references_typed():
    value = temporal_input()
    representation = TemporalRepresentation(value, representation_ref("t1-ref"), representation_ref("t2-ref"))
    assert representation.temporal_features == NOT_COMPUTED
    assert representation.change_features == NOT_COMPUTED
    with pytest.raises(ValueError, match="NOT_COMPUTED"):
        TemporalRepresentation(value, change_features="fake change")


def test_sequence_provenance_evidence_and_resource_estimation():
    sequence = TemporalSequenceInput((frame("one"), frame("two"), frame("three")), provenance("sequence"))
    evidence = TemporalEvidenceItem(TemporalEvidenceKind.OBSERVED_T1, "one", provenance("one"))
    estimate = ContractOnlyTemporalFusionAdapter().estimate_resources(temporal_input())
    assert len(sequence.frames) == 3 and evidence.kind is TemporalEvidenceKind.OBSERVED_T1
    assert estimate.estimated_image_count == 2 and estimate.estimated_vram_bytes == UNKNOWN
    with pytest.raises(ValueError, match="UNKNOWN"):
        TemporalResourceEstimate(2, estimated_vram_bytes="123")


def test_model_swappable_fusion_boundary_refuses_inference():
    adapter = ContractOnlyTemporalFusionAdapter()
    assert adapter.get_capabilities()["model_bound"] is False
    assert adapter.get_provenance()["model"] is None
    with pytest.raises(TemporalFusionUnavailable, match="not implemented"):
        adapter.encode_pair(temporal_input())
