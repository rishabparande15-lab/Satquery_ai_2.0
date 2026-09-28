from __future__ import annotations

import pytest

from src.spatial_contract import (
    BBox,
    CoordinateSpace,
    GeometryStatus,
    Point,
    RasterMetadata,
    analysis_to_normalized,
    analysis_to_token,
    geo_to_image,
    image_to_geo,
    normalized_to_pixel,
    pixel_to_normalized,
    pixel_to_token,
    spatial_fingerprint,
    token_index_to_row_col,
    token_row_col_to_index,
    token_to_analysis,
    validate_bbox,
    validate_point,
)


TRANSFORM = (10, 0, 373200, 0, -10, 5353200, 0, 0, 1)
METADATA = RasterMetadata(
    width=120, height=120, crs="EPSG:32633", transform=TRANSFORM,
    bounds=(373200, 5352000, 374400, 5353200), resolution=(10, 10),
)


def test_coordinate_spaces_and_half_open_bbox_contract():
    box = BBox(0, 0, 120, 120, CoordinateSpace.PIXEL)
    assert validate_bbox(box, width=120, height=120).status is GeometryStatus.VALID
    assert pixel_to_normalized(Point(60, 30, CoordinateSpace.PIXEL), width=120, height=120).to_dict() == {
        "x": 0.5, "y": 0.25, "coordinate_space": "NORMALIZED_IMAGE"}
    assert normalized_to_pixel(Point(0.5, 0.25, CoordinateSpace.NORMALIZED_IMAGE), width=120, height=120).x == 60


def test_validation_distinguishes_malformed_bounds_and_degenerate():
    assert validate_point(Point(-1, 0, CoordinateSpace.PIXEL), width=120, height=120).status is GeometryStatus.OUT_OF_BOUNDS
    assert validate_bbox(BBox(2, 2, 2, 5, CoordinateSpace.PIXEL), width=120, height=120).status is GeometryStatus.DEGENERATE
    assert validate_bbox(BBox(0, 0, 121, 5, CoordinateSpace.PIXEL), width=120, height=120).status is GeometryStatus.OUT_OF_BOUNDS
    assert validate_bbox({"x_min": 0}, width=120, height=120).status is GeometryStatus.MALFORMED
    with pytest.raises(ValueError, match="coordinate space"):
        pixel_to_normalized(Point(1, 1, CoordinateSpace.GEO), width=120, height=120)


def test_token_row_major_and_analysis_mapping():
    assert token_row_col_to_index(0, 0) == 0
    assert token_row_col_to_index(14, 14) == 224
    assert token_index_to_row_col(112) == (7, 7)
    assert analysis_to_token(Point(0, 0, CoordinateSpace.ANALYSIS_GRID)) == 0
    assert analysis_to_token(Point(119.99, 119.99, CoordinateSpace.ANALYSIS_GRID)) == 224
    assert pixel_to_token(Point(56, 56, CoordinateSpace.PIXEL), width=120, height=120) == 112
    assert token_to_analysis(224).to_dict() == {
        "x_min": 112.0, "y_min": 112.0, "x_max": 120.0, "y_max": 120.0,
        "coordinate_space": "ANALYSIS_GRID"}
    with pytest.raises(ValueError, match="out of range"):
        token_index_to_row_col(225)


def test_analysis_normalized_round_trip_is_deterministic():
    point = Point(12, 96, CoordinateSpace.ANALYSIS_GRID)
    normalized = analysis_to_normalized(point)
    assert normalized.to_dict() == {"x": 0.1, "y": 0.8, "coordinate_space": "NORMALIZED_IMAGE"}
    assert spatial_fingerprint(normalized.to_dict()) == spatial_fingerprint(normalized.to_dict())


def test_geo_transform_requires_metadata_and_round_trips():
    geo = image_to_geo(Point(0, 0, CoordinateSpace.PIXEL), METADATA)
    assert geo.to_dict() == {"x": 373200.0, "y": 5353200.0, "coordinate_space": "GEO"}
    assert geo_to_image(geo, METADATA).to_dict() == {
        "x": 0.0, "y": 0.0, "coordinate_space": "PIXEL"}
    with pytest.raises(ValueError, match="CRS and affine"):
        image_to_geo(Point(1, 1, CoordinateSpace.PIXEL), RasterMetadata(120, 120))


def test_raster_metadata_rejects_swapped_or_bad_orientation():
    with pytest.raises(ValueError, match="north-up"):
        RasterMetadata(120, 120, transform=(10, 0, 0, 0, 10, 1200))
    with pytest.raises(ValueError, match="positive"):
        RasterMetadata(0, 120)
    with pytest.raises(ValueError, match="bounds"):
        RasterMetadata(120, 120, bounds=(0, 0, 1, 0))
