"""Fail-closed contracts for future remote-sensing visual grounding.

These are metadata and validation boundaries only.  They do not convert source
geometry, map BigEarthNet.txt annotations, predict regions, or create evidence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from ..spatial_contract import BBox, CoordinateSpace, GeometryStatus, Point, validate_bbox, validate_point


GROUNDING_CONTRACT_VERSION = "phase3k2_grounding_contract_v1"
UNKNOWN = "UNKNOWN"


class RegionTargetType(str, Enum):
    POINT = "POINT"
    BBOX = "BBOX"
    MASK = "MASK"
    POLYGON = "POLYGON"
    REGION = "REGION"


class GroundingStatus(str, Enum):
    VERIFIED = "GROUNDING_VERIFIED"
    UNVERIFIED = "GROUNDING_UNVERIFIED"
    BLOCKED = "GROUNDING_BLOCKED"
    MODEL_UNAVAILABLE = "GROUNDING_MODEL_UNAVAILABLE"


class MappingStatus(str, Enum):
    VERIFIED = "MAPPING_VERIFIED"
    UNVERIFIED = "MAPPING_UNVERIFIED"


class GeometryOrigin(str, Enum):
    SOURCE_GEOMETRY = "SOURCE_GEOMETRY"
    LEARNED_PREDICTION = "LEARNED_PREDICTION"


class GroundingEvidenceKind(str, Enum):
    SOURCE_GEOMETRY = "SOURCE_GEOMETRY"
    MODEL_GROUNDING = "MODEL_GROUNDING"
    PIXEL_EVIDENCE = "PIXEL_EVIDENCE"
    REGION_EVIDENCE = "REGION_EVIDENCE"
    TOKEN_EVIDENCE = "TOKEN_EVIDENCE"
    METADATA_EVIDENCE = "METADATA_EVIDENCE"


class ConfidenceSource(str, Enum):
    MODEL_PROVIDED = "MODEL_PROVIDED"
    CALIBRATED = "CALIBRATED"
    UNKNOWN = "UNKNOWN"


class GroundingModelUnavailable(RuntimeError):
    """Raised by the contract-only adapter instead of fabricating geometry."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class ImageDimensions:
    width: int
    height: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive integers")


@dataclass(frozen=True)
class Polygon:
    points: tuple[tuple[float, float], ...]
    coordinate_space: CoordinateSpace | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        points = tuple(tuple(point) for point in self.points)
        if len(points) < 3 or any(len(point) != 2 for point in points):
            raise ValueError("polygon requires at least three x/y points")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   for point in points for value in point):
            raise ValueError("polygon coordinates must be finite")
        object.__setattr__(self, "points", tuple((float(x), float(y)) for x, y in points))


@dataclass(frozen=True)
class MaskReference:
    reference_id: str
    dimensions: ImageDimensions
    coordinate_space: CoordinateSpace | str
    provenance: Mapping[str, Any]
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _text(self.reference_id, "mask reference_id"))
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        if not isinstance(self.dimensions, ImageDimensions):
            raise ValueError("mask dimensions must be ImageDimensions")
        object.__setattr__(self, "provenance", _mapping(self.provenance, "mask provenance"))
        if self.sha256 is not None and (not isinstance(self.sha256, str) or len(self.sha256) != 64):
            raise ValueError("mask sha256 must be a 64-character digest when supplied")


Geometry = Point | BBox | Polygon | MaskReference


def validate_polygon(polygon: Polygon | Any, dimensions: ImageDimensions | None = None) -> GeometryStatus:
    """Return a status without clipping or otherwise changing polygon vertices."""
    if not isinstance(polygon, Polygon):
        return GeometryStatus.MALFORMED
    area = abs(sum(polygon.points[index][0] * polygon.points[(index + 1) % len(polygon.points)][1]
                   - polygon.points[(index + 1) % len(polygon.points)][0] * polygon.points[index][1]
                   for index in range(len(polygon.points)))) / 2.0
    if area == 0:
        return GeometryStatus.DEGENERATE
    if dimensions is not None and polygon.coordinate_space in {CoordinateSpace.PIXEL, CoordinateSpace.ANALYSIS_GRID}:
        if any(x < 0 or y < 0 or x > dimensions.width or y > dimensions.height for x, y in polygon.points):
            return GeometryStatus.OUT_OF_BOUNDS
    if dimensions is not None and polygon.coordinate_space is CoordinateSpace.NORMALIZED_IMAGE:
        if any(x < 0 or y < 0 or x > 1 or y > 1 for x, y in polygon.points):
            return GeometryStatus.OUT_OF_BOUNDS
    return GeometryStatus.VALID


def validate_geometry(geometry: Geometry, dimensions: ImageDimensions | None) -> GeometryStatus:
    if isinstance(geometry, Point):
        if dimensions is None:
            return GeometryStatus.MALFORMED
        width, height = (1, 1) if geometry.coordinate_space is CoordinateSpace.NORMALIZED_IMAGE else (dimensions.width, dimensions.height)
        return validate_point(geometry, width=width, height=height).status
    if isinstance(geometry, BBox):
        if dimensions is None:
            return GeometryStatus.MALFORMED
        width, height = (1, 1) if geometry.coordinate_space is CoordinateSpace.NORMALIZED_IMAGE else (dimensions.width, dimensions.height)
        return validate_bbox(geometry, width=width, height=height).status
    if isinstance(geometry, Polygon):
        return validate_polygon(geometry, dimensions)
    if isinstance(geometry, MaskReference):
        return GeometryStatus.VALID if dimensions is None or geometry.dimensions == dimensions else GeometryStatus.MALFORMED
    return GeometryStatus.MALFORMED


@dataclass(frozen=True)
class GroundingRequest:
    request_id: str
    image_id: str
    query_text: str
    modality: str
    target_type: RegionTargetType | str
    coordinate_space: CoordinateSpace | str
    provenance: Mapping[str, Any]
    image_dimensions: ImageDimensions | None = None
    annotation_id: str | None = None
    image_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("request_id", "image_id", "query_text", "modality"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "target_type", RegionTargetType(self.target_type))
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        if self.annotation_id is not None:
            object.__setattr__(self, "annotation_id", _text(self.annotation_id, "annotation_id"))
        if self.image_dimensions is not None and not isinstance(self.image_dimensions, ImageDimensions):
            raise ValueError("image_dimensions must be ImageDimensions when supplied")
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))
        object.__setattr__(self, "image_metadata", _mapping(self.image_metadata, "image_metadata"))


@dataclass(frozen=True)
class GroundingResult:
    request: GroundingRequest
    status: GroundingStatus | str
    geometry_origin: GeometryOrigin | str
    geometry: Geometry | None = None
    confidence: float | str = UNKNOWN
    confidence_source: ConfidenceSource | str = ConfidenceSource.UNKNOWN
    evidence_reference: Mapping[str, Any] | None = None
    model_provenance: Mapping[str, Any] = field(default_factory=dict)
    mapping_status: MappingStatus | str = MappingStatus.UNVERIFIED
    validation_status: GeometryStatus | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, GroundingRequest):
            raise ValueError("request must be GroundingRequest")
        object.__setattr__(self, "status", GroundingStatus(self.status))
        object.__setattr__(self, "geometry_origin", GeometryOrigin(self.geometry_origin))
        object.__setattr__(self, "mapping_status", MappingStatus(self.mapping_status))
        object.__setattr__(self, "model_provenance", _mapping(self.model_provenance, "model_provenance"))
        if self.evidence_reference is not None:
            object.__setattr__(self, "evidence_reference", _mapping(self.evidence_reference, "evidence_reference"))
        if self.geometry is not None and self.geometry_origin is not GeometryOrigin.LEARNED_PREDICTION:
            raise ValueError("a grounding result geometry must be a LEARNED_PREDICTION, not source geometry")
        expected_types = {Point: RegionTargetType.POINT, BBox: RegionTargetType.BBOX,
                          Polygon: RegionTargetType.POLYGON, MaskReference: RegionTargetType.MASK}
        if self.geometry is not None and self.request.target_type is not RegionTargetType.REGION:
            if expected_types.get(type(self.geometry)) is not self.request.target_type:
                raise ValueError("geometry type does not match requested region target type")
        status = validate_geometry(self.geometry, self.request.image_dimensions) if self.geometry is not None else GeometryStatus.MALFORMED
        if self.validation_status is not None and GeometryStatus(self.validation_status) is not status:
            raise ValueError("validation_status must match retained geometry validation")
        object.__setattr__(self, "validation_status", status)
        confidence_source = ConfidenceSource(self.confidence_source)
        object.__setattr__(self, "confidence_source", confidence_source)
        if confidence_source is ConfidenceSource.UNKNOWN:
            if self.confidence != UNKNOWN:
                raise ValueError("unknown confidence source requires UNKNOWN confidence")
        else:
            if (not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool)
                    or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
                raise ValueError("model-provided confidence must be a finite value in [0,1]")
        if self.status is GroundingStatus.VERIFIED and (status is not GeometryStatus.VALID or self.mapping_status is not MappingStatus.VERIFIED):
            raise ValueError("verified grounding requires valid geometry and verified mapping")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("status", "geometry_origin", "mapping_status", "confidence_source", "validation_status"):
            value[key] = getattr(self, key).value
        value["geometry"] = _geometry_dict(self.geometry)
        return value


def _geometry_dict(geometry: Geometry | None) -> dict[str, Any] | None:
    if geometry is None:
        return None
    if isinstance(geometry, (Point, BBox)):
        return geometry.to_dict()
    if isinstance(geometry, Polygon):
        return {"points": [list(point) for point in geometry.points], "coordinate_space": geometry.coordinate_space.value}
    return {"reference_id": geometry.reference_id, "dimensions": asdict(geometry.dimensions),
            "coordinate_space": geometry.coordinate_space.value, "sha256": geometry.sha256,
            "provenance": dict(geometry.provenance)}


class GroundingAdapter(ABC):
    """Model-neutral future boundary; it must not silently return a region."""

    @abstractmethod
    def validate_request(self, request: GroundingRequest) -> None: ...

    @abstractmethod
    def prepare_input(self, request: GroundingRequest) -> Mapping[str, Any]: ...

    @abstractmethod
    def predict(self, prepared: Mapping[str, Any]) -> GroundingResult: ...

    @abstractmethod
    def validate_output(self, result: GroundingResult) -> None: ...

    @abstractmethod
    def get_capabilities(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def get_provenance(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def estimate_resources(self, request: GroundingRequest) -> Mapping[str, Any]: ...


class ContractOnlyGroundingAdapter(GroundingAdapter):
    """Inspectable adapter that intentionally refuses learned grounding."""

    def validate_request(self, request: GroundingRequest) -> None:
        if not isinstance(request, GroundingRequest):
            raise ValueError("request must be GroundingRequest")

    def prepare_input(self, request: GroundingRequest) -> Mapping[str, Any]:
        self.validate_request(request)
        return {"request_id": request.request_id, "status": GroundingStatus.MODEL_UNAVAILABLE.value}

    def predict(self, prepared: Mapping[str, Any]) -> GroundingResult:
        raise GroundingModelUnavailable("learned grounding model is not implemented")

    def validate_output(self, result: GroundingResult) -> None:
        if not isinstance(result, GroundingResult):
            raise ValueError("result must be GroundingResult")
        if result.validation_status is not GeometryStatus.VALID:
            raise ValueError("invalid grounding geometry is rejected without clipping")
        if result.mapping_status is not MappingStatus.VERIFIED:
            raise ValueError("unverified mapping cannot be emitted as grounded evidence")

    def get_capabilities(self) -> Mapping[str, Any]:
        return {"learned_grounding": False, "qwen_grounding": "NOT_VERIFIED",
                "source_geometry_mapping": MappingStatus.UNVERIFIED.value,
                "modalities": {"OPTICAL": "ADAPTER_REQUIRED", "MULTISPECTRAL": "ADAPTER_REQUIRED",
                               "SAR": "ADAPTER_REQUIRED", "OPTICAL_SAR": "NOT_IMPLEMENTED", "TEMPORAL": "NOT_IMPLEMENTED"}}

    def get_provenance(self) -> Mapping[str, Any]:
        return {"contract_version": GROUNDING_CONTRACT_VERSION, "implementation": "contract_only", "model": None}

    def estimate_resources(self, request: GroundingRequest) -> Mapping[str, Any]:
        self.validate_request(request)
        return {"estimated_vram_bytes": UNKNOWN, "estimated_image_count": 1,
                "estimated_region_count": UNKNOWN, "estimated_mask_cost": UNKNOWN}
