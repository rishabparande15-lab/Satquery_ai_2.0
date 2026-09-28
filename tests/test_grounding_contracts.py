import pytest

from src.eo_vlm.grounding_contracts import (
    UNKNOWN, ConfidenceSource, ContractOnlyGroundingAdapter, GeometryOrigin,
    GroundingModelUnavailable, GroundingRequest, GroundingResult, GroundingStatus,
    ImageDimensions, MappingStatus, MaskReference, Polygon, RegionTargetType,
)
from src.spatial_contract import BBox, CoordinateSpace, GeometryStatus, Point


def request(**overrides):
    values = dict(request_id="ground-1", image_id="image-1", query_text="the runway", modality="OPTICAL",
                  target_type=RegionTargetType.BBOX, coordinate_space=CoordinateSpace.PIXEL,
                  image_dimensions=ImageDimensions(120, 120), provenance={"dataset_id": "fixture", "revision": "pinned"})
    values.update(overrides)
    return GroundingRequest(**values)


def result(geometry, **overrides):
    values = dict(request=request(), status=GroundingStatus.UNVERIFIED,
                  geometry_origin=GeometryOrigin.LEARNED_PREDICTION, geometry=geometry,
                  mapping_status=MappingStatus.UNVERIFIED)
    values.update(overrides)
    return GroundingResult(**values)


def test_request_requires_declared_coordinate_space_and_provenance():
    assert request().coordinate_space is CoordinateSpace.PIXEL
    with pytest.raises(ValueError, match="CoordinateSpace"):
        request(coordinate_space="UNDECLARED")
    with pytest.raises(ValueError, match="provenance"):
        request(provenance=None)


def test_point_and_bbox_validation_retains_invalid_geometry_without_clipping():
    valid = result(Point(10, 11, CoordinateSpace.PIXEL), request=request(target_type=RegionTargetType.POINT))
    invalid = result(BBox(-2, 4, 12, 18, CoordinateSpace.PIXEL))
    assert valid.validation_status is GeometryStatus.VALID
    assert invalid.validation_status is GeometryStatus.OUT_OF_BOUNDS
    assert invalid.geometry.x_min == -2
    with pytest.raises(ValueError, match="verified grounding"):
        result(invalid.geometry, status=GroundingStatus.VERIFIED, mapping_status=MappingStatus.VERIFIED)


def test_polygon_and_mask_reference_validation():
    polygon = Polygon(((0, 0), (10, 0), (0, 10)), CoordinateSpace.PIXEL)
    assert result(polygon, request=request(target_type=RegionTargetType.POLYGON)).validation_status is GeometryStatus.VALID
    degenerate = Polygon(((0, 0), (1, 1), (2, 2)), CoordinateSpace.PIXEL)
    assert result(degenerate, request=request(target_type=RegionTargetType.POLYGON)).validation_status is GeometryStatus.DEGENERATE
    mask = MaskReference("mask://one", ImageDimensions(120, 120), CoordinateSpace.PIXEL, {"source": "verified"}, "a" * 64)
    assert result(mask, request=request(target_type=RegionTargetType.MASK)).validation_status is GeometryStatus.VALID
    with pytest.raises(ValueError, match="64-character"):
        MaskReference("mask://bad", ImageDimensions(1, 1), CoordinateSpace.PIXEL, {}, "bad")


def test_source_geometry_cannot_be_recast_as_learned_prediction():
    with pytest.raises(ValueError, match="not source geometry"):
        GroundingResult(request(), GroundingStatus.UNVERIFIED, GeometryOrigin.SOURCE_GEOMETRY,
                        BBox(1, 1, 2, 2, CoordinateSpace.PIXEL))


def test_phase2f_and_token_mapping_stay_fail_closed():
    adapter = ContractOnlyGroundingAdapter()
    capabilities = adapter.get_capabilities()
    assert capabilities["source_geometry_mapping"] == "MAPPING_UNVERIFIED"
    assert capabilities["modalities"]["TEMPORAL"] == "NOT_IMPLEMENTED"
    with pytest.raises(ValueError, match="unverified mapping"):
        adapter.validate_output(result(BBox(1, 1, 2, 2, CoordinateSpace.PIXEL)))


def test_confidence_requires_explicit_source_and_model_swappable_adapter_refuses_prediction():
    with pytest.raises(ValueError, match="unknown confidence"):
        result(BBox(1, 1, 2, 2, CoordinateSpace.PIXEL), confidence=0.5)
    observed = result(BBox(1, 1, 2, 2, CoordinateSpace.PIXEL), confidence=0.5,
                      confidence_source=ConfidenceSource.MODEL_PROVIDED)
    assert observed.confidence == 0.5
    adapter = ContractOnlyGroundingAdapter()
    assert adapter.estimate_resources(request())["estimated_vram_bytes"] == UNKNOWN
    with pytest.raises(GroundingModelUnavailable, match="not implemented"):
        adapter.predict(adapter.prepare_input(request()))
