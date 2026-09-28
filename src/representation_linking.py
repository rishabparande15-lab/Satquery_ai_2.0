"""Split-safe links from verified annotations to existing representations.

This module creates geometric correspondence and provenance links only. It does
not infer semantics and never calls geometric correspondence learned grounding.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .architecture_contracts import RepresentationRef
from .spatial_contract import (
    BBox,
    CoordinateSpace,
    Point,
    analysis_to_normalized,
    bbox_normalized_to_pixel,
    pixel_to_token,
    token_index_to_row_col,
    token_to_analysis,
    validate_bbox,
    validate_point,
)

SCHEMA_VERSION = "representation_link_v1"
LINKAGE_VERSION = "annotation_representation_link_v1"


class LinkStatus(str, Enum):
    LINKED = "LINKED"
    REPRESENTATION_MISSING = "REPRESENTATION_MISSING"
    SPATIAL_LINK_UNAVAILABLE = "SPATIAL_LINK_UNAVAILABLE"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    SPLIT_MISMATCH = "SPLIT_MISMATCH"
    SPLIT_UNKNOWN = "SPLIT_UNKNOWN"
    IMAGE_UNMATCHED = "IMAGE_UNMATCHED"


class RelationshipType(str, Enum):
    IMAGE_TO_REPRESENTATION = "IMAGE_TO_REPRESENTATION"
    BBOX_TO_ANALYSIS_REGION = "BBOX_TO_ANALYSIS_REGION"
    BBOX_TO_CROMA_TOKENS = "BBOX_TO_CROMA_TOKENS"
    POINT_TO_ANALYSIS_PIXEL = "POINT_TO_ANALYSIS_PIXEL"
    POINT_TO_CROMA_TOKEN = "POINT_TO_CROMA_TOKEN"


@dataclass(frozen=True)
class AnnotationRepresentationLink:
    link_id: str
    annotation_id: str
    image_id: str | None
    split: str | None
    annotation_type: str | None
    task_type: str | None
    spatial_reference: Mapping[str, Any] | None
    representation_refs: tuple[Mapping[str, Any], ...]
    representation_availability: Mapping[str, str]
    relationship_types: tuple[str, ...]
    evidence_reference: Mapping[str, Any] | None
    source_provenance: Mapping[str, Any]
    linkage_version: str = LINKAGE_VERSION
    validation_status: str = "valid"
    validation_issues: tuple[str, ...] = ()
    scene_id: str | None = None
    observation_id: str | None = None
    temporal_group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def link_annotation(
    annotation: Mapping[str, Any],
    image_manifest: Mapping[str, Mapping[str, Any]],
    representations: Iterable[Mapping[str, Any] | RepresentationRef] = (),
    *,
    canonical_geometry: BBox | Point | None = None,
    evidence_reference: Mapping[str, Any] | None = None,
    requested_representation_types: Iterable[str] = (),
) -> AnnotationRepresentationLink:
    """Create one fail-closed link from an annotation to trusted image data."""
    annotation_id = annotation.get("annotation_id")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError("annotation_id is required")
    image_id = annotation.get("image_id")
    image = image_manifest.get(image_id) if isinstance(image_id, str) else None
    issues: list[str] = []
    relationships: list[str] = []
    spatial_reference: dict[str, Any] | None = None
    rep_values = [_representation_dict(value) for value in representations]
    availability: dict[str, str] = {str(kind): "REPRESENTATION_MISSING" for kind in requested_representation_types}

    if image is None:
        issues.append(LinkStatus.IMAGE_UNMATCHED.value)
    manifest_split = image.get("split") if image else None
    annotation_split = annotation.get("split")
    split = manifest_split if isinstance(manifest_split, str) else annotation_split if isinstance(annotation_split, str) else None
    if image is not None and annotation_split is not None and manifest_split != annotation_split:
        issues.append(LinkStatus.SPLIT_MISMATCH.value)
    if image is not None and manifest_split is None:
        issues.append(LinkStatus.SPLIT_UNKNOWN.value)

    linked_reps: list[dict[str, Any]] = []
    for representation in rep_values:
        rep_type = str(representation.get("representation_type", representation.get("type", "unknown")))
        availability[rep_type] = "available"
        rep_scene = representation.get("scene_id", representation.get("image_id"))
        if image_id is not None and rep_scene != image_id:
            issues.append(LinkStatus.PROVENANCE_MISMATCH.value)
            availability[rep_type] = LinkStatus.PROVENANCE_MISMATCH.value
            continue
        rep_split = representation.get("split")
        if split is not None and rep_split is None:
            issues.append(LinkStatus.SPLIT_UNKNOWN.value)
        elif split is not None and rep_split != split:
            issues.append(LinkStatus.SPLIT_MISMATCH.value)
            availability[rep_type] = LinkStatus.SPLIT_MISMATCH.value
            continue
        linked_reps.append(_canonical_representation(representation))

    if canonical_geometry is not None:
        if image is None:
            issues.append(LinkStatus.SPATIAL_LINK_UNAVAILABLE.value)
        else:
            try:
                spatial_reference, geometry_relationships = _spatial_link(canonical_geometry)
                spatial_reference["source_geometry"] = _source_geometry(annotation)
                spatial_reference["source_coordinate_semantics"] = annotation.get("geometry_frame")
                relationships.extend(geometry_relationships)
            except ValueError as error:
                spatial_reference = {"status": LinkStatus.SPATIAL_LINK_UNAVAILABLE.value,
                                     "reason": str(error),
                                     "source_geometry": _source_geometry(annotation)}
                issues.append(LinkStatus.SPATIAL_LINK_UNAVAILABLE.value)
    elif annotation.get("box") is not None or annotation.get("point") is not None:
        spatial_reference = {"status": LinkStatus.SPATIAL_LINK_UNAVAILABLE.value,
                             "reason": "source geometry has no validated canonical coordinate space",
                             "source_geometry": _source_geometry(annotation),
                             "source_coordinate_semantics": annotation.get("geometry_frame")}
        issues.append(LinkStatus.SPATIAL_LINK_UNAVAILABLE.value)

    if not linked_reps:
        issues.append(LinkStatus.REPRESENTATION_MISSING.value)
    if not relationships:
        relationships.append(RelationshipType.IMAGE_TO_REPRESENTATION.value)
    unique_issues = tuple(sorted(set(issues)))
    status = _status(unique_issues)
    link_key = [LINKAGE_VERSION, annotation_id, image_id, split,
                [rep.get("reference_id", rep.get("representation_id")) for rep in linked_reps], spatial_reference]
    link_id = f"arl:{sha256(_canonical_bytes(link_key)).hexdigest()}"
    return AnnotationRepresentationLink(
        link_id=link_id,
        annotation_id=annotation_id,
        image_id=image_id if isinstance(image_id, str) else None,
        split=split,
        annotation_type=annotation.get("annotation_type", annotation.get("task_type")),
        task_type=annotation.get("task_type"),
        spatial_reference=spatial_reference,
        representation_refs=tuple(linked_reps),
        representation_availability=dict(sorted(availability.items())),
        relationship_types=tuple(sorted(set(relationships))),
        evidence_reference=evidence_reference,
        source_provenance=dict(annotation.get("provenance") or {}),
        validation_status=status,
        validation_issues=unique_issues,
    )


def write_link_artifact(
    annotations: Iterable[Mapping[str, Any]],
    image_manifest: Mapping[str, Mapping[str, Any]],
    representations_by_image: Mapping[str, Iterable[Mapping[str, Any] | RepresentationRef]],
    output: Path,
    *,
    requested_representation_types: Iterable[str] = (),
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Write stable links and audit files without machine-specific paths."""
    links = [link_annotation(
        annotation,
        image_manifest,
        representations_by_image.get(annotation.get("image_id"), ()),
        requested_representation_types=requested_representation_types,
    ).to_dict() for annotation in annotations]
    links.sort(key=lambda value: value["link_id"])
    content = b"".join(_canonical_bytes(value) + b"\n" for value in links)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "links.jsonl").write_bytes(content)
    report = {
        "schema_version": SCHEMA_VERSION,
        "linkage_version": LINKAGE_VERSION,
        "link_count": len(links),
        "image_level_links": sum(RelationshipType.IMAGE_TO_REPRESENTATION.value in link["relationship_types"] for link in links),
        "bbox_links": sum(RelationshipType.BBOX_TO_CROMA_TOKENS.value in link["relationship_types"] for link in links),
        "point_links": sum(RelationshipType.POINT_TO_CROMA_TOKEN.value in link["relationship_types"] for link in links),
        "missing_representation_cases": sum(LinkStatus.REPRESENTATION_MISSING.value in link["validation_issues"] for link in links),
        "spatial_link_failures": sum(LinkStatus.SPATIAL_LINK_UNAVAILABLE.value in link["validation_issues"] for link in links),
        "split_failures": sum(issue in link["validation_issues"] for link in links for issue in (LinkStatus.SPLIT_MISMATCH.value, LinkStatus.SPLIT_UNKNOWN.value)),
        "provenance_failures": sum(LinkStatus.PROVENANCE_MISMATCH.value in link["validation_issues"] for link in links),
        "status_counts": dict(sorted(Counter(link["validation_status"] for link in links).items())),
        "link_sha256": sha256(content).hexdigest(),
    }
    manifest = {"schema_version": SCHEMA_VERSION, "linkage_version": LINKAGE_VERSION,
                "links_file": "links.jsonl", "links_sha256": report["link_sha256"],
                "link_count": len(links)}
    provenance = {"schema_version": SCHEMA_VERSION, "linkage_version": LINKAGE_VERSION,
                  "code_revision": code_revision, "pipeline3_modified": False,
                  "model_training_performed": False}
    (output / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
    (output / "validation_report.json").write_bytes(_canonical_bytes(report) + b"\n")
    (output / "provenance.json").write_bytes(_canonical_bytes(provenance) + b"\n")
    return report


def _spatial_link(geometry: BBox | Point) -> tuple[dict[str, Any], list[str]]:
    if isinstance(geometry, Point):
        validation = validate_point(geometry, width=120, height=120)
        if not validation.valid:
            raise ValueError(f"point {validation.status.value}: {validation.reason}")
        token = pixel_to_token(Point(geometry.x, geometry.y, CoordinateSpace.PIXEL), width=120, height=120) \
            if geometry.coordinate_space is CoordinateSpace.PIXEL else _analysis_point_to_token(geometry)
        row, col = token_index_to_row_col(token)
        return ({"status": "valid", "canonical": geometry.to_dict(),
                 "analysis_pixel": {"x": geometry.x, "y": geometry.y, "coordinate_space": geometry.coordinate_space.value},
                 "token_index": token, "token_row": row, "token_col": col,
                 "learned_grounding": False},
                [RelationshipType.POINT_TO_ANALYSIS_PIXEL.value, RelationshipType.POINT_TO_CROMA_TOKEN.value])
    if not isinstance(geometry, BBox):
        raise ValueError("canonical geometry must be a Point or BBox")
    if geometry.coordinate_space is CoordinateSpace.NORMALIZED_IMAGE:
        geometry = bbox_normalized_to_pixel(geometry, width=120, height=120)
    elif geometry.coordinate_space is not CoordinateSpace.PIXEL and geometry.coordinate_space is not CoordinateSpace.ANALYSIS_GRID:
        raise ValueError("bbox must use PIXEL, ANALYSIS_GRID, or NORMALIZED_IMAGE")
    validation = validate_bbox(geometry, width=120, height=120)
    if not validation.valid:
        raise ValueError(f"bbox {validation.status.value}: {validation.reason}")
    tokens = _bbox_tokens(geometry)
    return ({"status": "valid", "canonical": geometry.to_dict(),
             "analysis_region": geometry.to_dict(),
             "token_links": tokens, "learned_grounding": False},
            [RelationshipType.BBOX_TO_ANALYSIS_REGION.value, RelationshipType.BBOX_TO_CROMA_TOKENS.value])


def _bbox_tokens(box: BBox) -> list[dict[str, Any]]:
    result = []
    for index in range(225):
        token = token_to_analysis(index)
        left = max(box.x_min, token.x_min)
        top = max(box.y_min, token.y_min)
        right = min(box.x_max, token.x_max)
        bottom = min(box.y_max, token.y_max)
        if right <= left or bottom <= top:
            continue
        intersection = (right - left) * (bottom - top)
        token_area = (token.x_max - token.x_min) * (token.y_max - token.y_min)
        box_area = (box.x_max - box.x_min) * (box.y_max - box.y_min)
        if box.x_min <= token.x_min and box.y_min <= token.y_min and box.x_max >= token.x_max and box.y_max >= token.y_max:
            relationship = "FULLY_CONTAINED"
        elif token.x_min <= box.x_min and token.y_min <= box.y_min and token.x_max >= box.x_max and token.y_max >= box.y_max:
            relationship = "CONTAINS"
        else:
            relationship = "INTERSECTS"
        row, col = token_index_to_row_col(index)
        result.append({"token_index": index, "token_row": row, "token_col": col,
                       "relationship": relationship,
                       "intersection_fraction_of_token": intersection / token_area,
                       "intersection_fraction_of_bbox": intersection / box_area})
    return result


def _analysis_point_to_token(point: Point) -> int:
    from .spatial_contract import analysis_to_token
    return analysis_to_token(point)


def _source_geometry(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {"box": annotation.get("box", annotation.get("bbox")), "point": annotation.get("point")}


def _status(issues: tuple[str, ...]) -> str:
    serious = {LinkStatus.IMAGE_UNMATCHED.value, LinkStatus.SPLIT_MISMATCH.value,
               LinkStatus.PROVENANCE_MISMATCH.value, LinkStatus.SPATIAL_LINK_UNAVAILABLE.value}
    if serious.intersection(issues):
        return "invalid"
    if LinkStatus.SPLIT_UNKNOWN.value in issues or LinkStatus.REPRESENTATION_MISSING.value in issues:
        return "partial"
    return "valid"


def _representation_dict(value: Mapping[str, Any] | RepresentationRef) -> dict[str, Any]:
    if isinstance(value, RepresentationRef):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("representation must be a mapping or RepresentationRef")
    return dict(value)


def _canonical_representation(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"reference_id", "representation_id", "representation_type", "type", "scene_id", "image_id",
               "split", "modality", "version", "shape", "dtype", "producer", "producer_version",
               "preprocessing_version", "checksum_sha256", "artifact_id", "artifact_uri", "provenance_reference",
               "status", "spatial_reference"}
    result = {key: value[key] for key in sorted(value) if key in allowed}
    artifact = value.get("artifact")
    if isinstance(artifact, Mapping):
        result["artifact_id"] = artifact.get("artifact_id")
        result["artifact_uri"] = artifact.get("uri")
        result["checksum_sha256"] = artifact.get("checksum_sha256")
        result["provenance_reference"] = value.get("provenance_reference") or artifact.get("provenance_reference")
    return {key: result[key] for key in sorted(result) if result[key] is not None}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
