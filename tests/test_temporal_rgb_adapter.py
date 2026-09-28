import numpy as np
import pytest

from src.eo_vlm.temporal_contracts import (
    SpatialCorrespondence, SpatialCorrespondenceStatus, TemporalFrame, TemporalInput,
    TemporalMetadata, TemporalModality, TemporalOrderStatus, TemporalProvenance,
)
from src.eo_vlm.temporal_rgb_adapter import FIXED_TEMPORAL_RGB_PROMPT, TemporalRGBAdapter


def pair():
    source = TemporalProvenance("google/RSRCC", source_revision="7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea", attributes={"split": "val"})
    return TemporalInput(
        TemporalFrame("pair_before", TemporalModality.OPTICAL, source, tensor_shape=(512, 512, 3)),
        TemporalFrame("pair_after", TemporalModality.OPTICAL, source, tensor_shape=(512, 512, 3)),
        TemporalOrderStatus.VERIFIED, SpatialCorrespondenceStatus.VERIFIED,
        SpatialCorrespondence.SPATIALLY_CORRESPONDING, TemporalMetadata(), source,
    )


def images():
    return np.zeros((8, 9, 3), dtype=np.uint8), np.full((8, 9, 3), 7, dtype=np.uint8)


def test_rgb_pair_validation_identity_order_and_unknown_metadata():
    value = pair(); a, b = images()
    TemporalRGBAdapter().validate_temporal_rgb(value, a, b)
    prepared = TemporalRGBAdapter().prepare_temporal_rgb(value, a, b)
    assert prepared.provenance["t1_reference_id"] == "pair_before"
    assert prepared.provenance["temporal_order"] == "TEMPORAL_ORDER_VERIFIED"
    assert prepared.provenance["timestamp_status"] == "UNKNOWN"
    assert prepared.provenance["spatial_correspondence_status"] == "SPATIAL_CORRESPONDENCE_VERIFIED"
    assert prepared.provenance["registration_status"] == "UNKNOWN"


def test_preparation_is_deterministic_and_preserves_two_rgb_inputs():
    a, b = images(); adapter = TemporalRGBAdapter()
    one = adapter.prepare_temporal_rgb(pair(), a, b); two = adapter.prepare_temporal_rgb(pair(), a, b)
    assert one.model_input[0].shape == (8, 9, 3) and one.model_input[1].shape == (8, 9, 3)
    assert one.provenance["pair_input_sha256"] == two.provenance["pair_input_sha256"]
    assert one.prompt == FIXED_TEMPORAL_RGB_PROMPT


def test_malformed_or_non_optical_pair_is_rejected():
    a, b = images(); source = TemporalProvenance("source")
    bad = TemporalInput(TemporalFrame("a", TemporalModality.S1, source, tensor_shape=(2, 120, 120)),
                        TemporalFrame("b", TemporalModality.S1, source, tensor_shape=(2, 120, 120)),
                        TemporalOrderStatus.UNKNOWN, SpatialCorrespondenceStatus.UNKNOWN, provenance=source)
    with pytest.raises(ValueError, match="optical RGB"):
        TemporalRGBAdapter().validate_temporal_rgb(bad, a, b)
    with pytest.raises(ValueError, match="HxWx3"):
        TemporalRGBAdapter().validate_temporal_rgb(pair(), np.zeros((3, 8, 9)), b)


def test_capabilities_provenance_resources_and_model_swappable_interface():
    adapter = TemporalRGBAdapter(); estimate = adapter.estimate_resources(pair(), max_new_tokens=16)
    assert adapter.get_capabilities()["learned_temporal_fusion"] is False
    assert adapter.get_provenance()["scientific_representation_used"] is False
    assert estimate["estimated_image_count"] == 2 and estimate["temporal_fusion"] == "NOT_COMPUTED"
