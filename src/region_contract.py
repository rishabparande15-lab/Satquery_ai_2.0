"""Minimal source-region contract for deterministic future region consumers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

from .spatial_contract import BBox, Point, CoordinateSpace

SCHEMA_VERSION = "region_contract_v1"
SOURCE_ANNOTATION_REGION = "SOURCE_ANNOTATION_REGION"
DETERMINISTIC_REGION = "DETERMINISTIC_REGION"
LEARNED_REGION = "LEARNED_REGION"
_ALLOWED_SOURCES = {SOURCE_ANNOTATION_REGION, DETERMINISTIC_REGION}


@dataclass(frozen=True)
class Region:
    region_id: str
    image_id: str
    split: str
    coordinate_space: CoordinateSpace
    geometry: Mapping[str, Any]
    source: str
    linked_tokens: tuple[int, ...]
    linked_pixels: tuple[tuple[int, int], ...]
    provenance: Mapping[str, Any]
    validation_status: str = "valid"

    def __post_init__(self) -> None:
        if not self.region_id or not self.image_id or self.split not in {"train", "validation", "test"}:
            raise ValueError("region identity and split are required")
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        if self.source not in _ALLOWED_SOURCES:
            raise ValueError("learned regions are reserved and cannot be materialized in Phase 2D")
        if any(type(index) is not int or not 0 <= index < 225 for index in self.linked_tokens):
            raise ValueError("linked token index is invalid")
        if not isinstance(self.geometry, Mapping) or not isinstance(self.provenance, Mapping):
            raise ValueError("geometry and provenance must be mappings")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["coordinate_space"] = self.coordinate_space.value
        return value


def source_region(region_id: str, image_id: str, split: str, geometry: BBox | Point, *, provenance: Mapping[str, Any]) -> Region:
    return Region(region_id, image_id, split, geometry.coordinate_space, geometry.to_dict(), SOURCE_ANNOTATION_REGION, (), (), provenance)
