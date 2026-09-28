"""Authoritative, deterministic spatial/raster geometry contract.

This module describes geometry only. It does not resample rasters, infer CRS,
clip invalid annotations, or claim learned/geographic grounding when metadata is
missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Sequence

SCHEMA_VERSION = "spatial_contract_v1"
MAPPING_VERSION = "north_up_row_major_120_to_15_v1"
DEFAULT_ANALYSIS_SHAPE = (120, 120)
DEFAULT_TOKEN_GRID = (15, 15)


class CoordinateSpace(str, Enum):
    PIXEL = "PIXEL"
    ANALYSIS_GRID = "ANALYSIS_GRID"
    CROMA_TOKEN = "CROMA_TOKEN"
    NORMALIZED_IMAGE = "NORMALIZED_IMAGE"
    GEO = "GEO"


class GeometryStatus(str, Enum):
    VALID = "valid"
    OUT_OF_BOUNDS = "out_of_bounds"
    MALFORMED = "malformed"
    DEGENERATE = "degenerate"


@dataclass(frozen=True)
class GeometryValidation:
    status: GeometryStatus
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.status is GeometryStatus.VALID


@dataclass(frozen=True)
class RasterMetadata:
    """Metadata needed for deterministic image/analysis/GEO transforms."""

    width: int
    height: int
    crs: str | None = None
    transform: tuple[float, ...] | None = None
    bounds: tuple[float, float, float, float] | None = None
    resolution: tuple[float, float] | None = None
    role: str = "analysis_grid"

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0:
            raise ValueError("Raster width and height must be positive integers")
        if self.transform is not None:
            transform = tuple(float(value) for value in self.transform)
            if len(transform) not in (6, 9) or not all(math.isfinite(value) for value in transform):
                raise ValueError("Raster transform must be six or nine finite values")
            if len(transform) == 9 and transform[6:] != (0.0, 0.0, 1.0):
                raise ValueError("Raster transform must be an affine six-term or homogeneous transform")
            a, b, c, d, e, f = transform[:6]
            if a <= 0 or e >= 0 or b != 0 or d != 0:
                raise ValueError("Only north-up transforms with positive x and negative y scale are supported")
            object.__setattr__(self, "transform", transform)
        if self.bounds is not None:
            bounds = tuple(float(value) for value in self.bounds)
            if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError("Raster bounds must be finite left, bottom, right, top values")
            object.__setattr__(self, "bounds", bounds)
        if self.resolution is not None:
            resolution = tuple(float(value) for value in self.resolution)
            if len(resolution) != 2 or not all(math.isfinite(value) and value > 0 for value in resolution):
                raise ValueError("Raster resolution must contain two positive finite values")
            object.__setattr__(self, "resolution", resolution)
        if self.crs is not None and (not isinstance(self.crs, str) or not self.crs.strip()):
            raise ValueError("CRS must be a non-empty string when supplied")


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    coordinate_space: CoordinateSpace

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (self.x, self.y)):
            raise ValueError("Point coordinates must be finite numbers")

    def to_dict(self) -> dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y), "coordinate_space": self.coordinate_space.value}


@dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    coordinate_space: CoordinateSpace

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite numbers")

    def to_dict(self) -> dict[str, Any]:
        return {"x_min": float(self.x_min), "y_min": float(self.y_min),
                "x_max": float(self.x_max), "y_max": float(self.y_max),
                "coordinate_space": self.coordinate_space.value}


def _shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Shape must be two positive integers in (height, width) order")
    return int(shape[0]), int(shape[1])


def _space(value: CoordinateSpace | str) -> CoordinateSpace:
    try:
        return CoordinateSpace(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported coordinate space: {value!r}") from error


def validate_point(point: Point | Any, *, width: int, height: int, space: CoordinateSpace | str | None = None) -> GeometryValidation:
    if not isinstance(point, Point):
        return GeometryValidation(GeometryStatus.MALFORMED, "point must be a Point")
    if space is not None and point.coordinate_space is not _space(space):
        return GeometryValidation(GeometryStatus.MALFORMED, "point coordinate space does not match expected space")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        return GeometryValidation(GeometryStatus.MALFORMED, "invalid raster dimensions")
    if not (0 <= point.x <= width and 0 <= point.y <= height):
        return GeometryValidation(GeometryStatus.OUT_OF_BOUNDS, "point is outside the image boundary")
    return GeometryValidation(GeometryStatus.VALID)


def validate_bbox(box: BBox | Any, *, width: int, height: int, space: CoordinateSpace | str | None = None) -> GeometryValidation:
    if not isinstance(box, BBox):
        return GeometryValidation(GeometryStatus.MALFORMED, "bbox must be a BBox")
    if space is not None and box.coordinate_space is not _space(space):
        return GeometryValidation(GeometryStatus.MALFORMED, "bbox coordinate space does not match expected space")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        return GeometryValidation(GeometryStatus.MALFORMED, "invalid raster dimensions")
    if box.x_max <= box.x_min or box.y_max <= box.y_min:
        return GeometryValidation(GeometryStatus.DEGENERATE, "bbox must have positive width and height")
    if box.x_min < 0 or box.y_min < 0 or box.x_max > width or box.y_max > height:
        return GeometryValidation(GeometryStatus.OUT_OF_BOUNDS, "bbox exceeds the image boundary")
    return GeometryValidation(GeometryStatus.VALID)


def pixel_to_normalized(point: Point, *, width: int, height: int) -> Point:
    _require_point(point, CoordinateSpace.PIXEL, width, height)
    return Point(point.x / width, point.y / height, CoordinateSpace.NORMALIZED_IMAGE)


def normalized_to_pixel(point: Point, *, width: int, height: int) -> Point:
    _require_point(point, CoordinateSpace.NORMALIZED_IMAGE, 1, 1)
    return Point(point.x * width, point.y * height, CoordinateSpace.PIXEL)


def bbox_pixel_to_normalized(box: BBox, *, width: int, height: int) -> BBox:
    _require_bbox(box, CoordinateSpace.PIXEL, width, height)
    return BBox(box.x_min / width, box.y_min / height, box.x_max / width, box.y_max / height, CoordinateSpace.NORMALIZED_IMAGE)


def bbox_normalized_to_pixel(box: BBox, *, width: int, height: int) -> BBox:
    _require_bbox(box, CoordinateSpace.NORMALIZED_IMAGE, 1, 1)
    return BBox(box.x_min * width, box.y_min * height, box.x_max * width, box.y_max * height, CoordinateSpace.PIXEL)


def analysis_to_normalized(point: Point, *, width: int = 120, height: int = 120) -> Point:
    _require_point(point, CoordinateSpace.ANALYSIS_GRID, width, height)
    return Point(point.x / width, point.y / height, CoordinateSpace.NORMALIZED_IMAGE)


def normalized_to_analysis(point: Point, *, width: int = 120, height: int = 120) -> Point:
    _require_point(point, CoordinateSpace.NORMALIZED_IMAGE, 1, 1)
    return Point(point.x * width, point.y * height, CoordinateSpace.ANALYSIS_GRID)


def token_row_col_to_index(row: int, col: int, *, grid: tuple[int, int] = DEFAULT_TOKEN_GRID) -> int:
    rows, cols = grid
    if type(row) is not int or type(col) is not int or not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("Token row/column is out of range")
    return row * cols + col


def token_index_to_row_col(index: int, *, grid: tuple[int, int] = DEFAULT_TOKEN_GRID) -> tuple[int, int]:
    rows, cols = grid
    if type(index) is not int or not 0 <= index < rows * cols:
        raise ValueError("Token index is out of range")
    return divmod(index, cols)


def analysis_to_token(point: Point, *, width: int = 120, height: int = 120, grid: tuple[int, int] = DEFAULT_TOKEN_GRID) -> int:
    _require_point(point, CoordinateSpace.ANALYSIS_GRID, width, height)
    rows, cols = grid
    if width % cols or height % rows:
        raise ValueError("Analysis dimensions must be divisible by token grid")
    col = min(cols - 1, int(point.x // (width / cols)))
    row = min(rows - 1, int(point.y // (height / rows)))
    return token_row_col_to_index(row, col, grid=grid)


def token_to_analysis(index: int, *, width: int = 120, height: int = 120, grid: tuple[int, int] = DEFAULT_TOKEN_GRID) -> BBox:
    row, col = token_index_to_row_col(index, grid=grid)
    rows, cols = grid
    if width % cols or height % rows:
        raise ValueError("Analysis dimensions must be divisible by token grid")
    block_width, block_height = width / cols, height / rows
    return BBox(col * block_width, row * block_height, (col + 1) * block_width,
                (row + 1) * block_height, CoordinateSpace.ANALYSIS_GRID)


def pixel_to_token(point: Point, *, width: int, height: int, grid: tuple[int, int] = DEFAULT_TOKEN_GRID) -> int:
    _require_point(point, CoordinateSpace.PIXEL, width, height)
    return analysis_to_token(Point(point.x, point.y, CoordinateSpace.ANALYSIS_GRID), width=width, height=height, grid=grid)


def token_to_pixel(index: int, *, width: int, height: int, grid: tuple[int, int] = DEFAULT_TOKEN_GRID) -> BBox:
    box = token_to_analysis(index, width=width, height=height, grid=grid)
    return BBox(box.x_min, box.y_min, box.x_max, box.y_max, CoordinateSpace.PIXEL)


def image_to_geo(point: Point, metadata: RasterMetadata) -> Point:
    _require_point(point, CoordinateSpace.PIXEL, metadata.width, metadata.height)
    if metadata.crs is None or metadata.transform is None:
        raise ValueError("GEO transform requires validated CRS and affine transform")
    x, y = point.x, point.y
    a, b, c, d, e, f = metadata.transform[:6]
    return Point(a * x + b * y + c, d * x + e * y + f, CoordinateSpace.GEO)


def geo_to_image(point: Point, metadata: RasterMetadata) -> Point:
    if point.coordinate_space is not CoordinateSpace.GEO:
        raise ValueError("geo_to_image requires a GEO point")
    if metadata.crs is None or metadata.transform is None:
        raise ValueError("GEO transform requires validated CRS and affine transform")
    a, b, c, d, e, f = metadata.transform[:6]
    determinant = a * e - b * d
    if determinant == 0:
        raise ValueError("Affine transform is not invertible")
    dx, dy = point.x - c, point.y - f
    return Point((e * dx - b * dy) / determinant, (-d * dx + a * dy) / determinant, CoordinateSpace.PIXEL)


def spatial_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_point(point: Point, space: CoordinateSpace, width: int, height: int) -> None:
    result = validate_point(point, width=width, height=height, space=space)
    if not result.valid:
        raise ValueError(f"Invalid {space.value} point: {result.status.value}: {result.reason}")


def _require_bbox(box: BBox, space: CoordinateSpace, width: int, height: int) -> None:
    result = validate_bbox(box, width=width, height=height, space=space)
    if not result.valid:
        raise ValueError(f"Invalid {space.value} bbox: {result.status.value}: {result.reason}")
