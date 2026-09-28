from __future__ import annotations

import json

import pytest

from src.representation_linking import (
    LinkStatus,
    RelationshipType,
    link_annotation,
    write_link_artifact,
)
from src.spatial_contract import BBox, CoordinateSpace, Point


IMAGES = {"area-1": {"split": "train"}, "area-2": {"split": "test"}}
REPRESENTATIONS = [
    {"reference_id": "joint-1", "representation_type": "joint_croma", "scene_id": "area-1",
     "split": "train", "producer": "CROMA", "producer_version": "checkpoint-v1",
     "preprocessing_version": "croma-readme-v1", "checksum_sha256": "a" * 64,
     "artifact_uri": "artifact://representations/area-1/joint.npy", "spatial_reference": {"level": "token"}},
]


def annotation(**changes):
    value = {"annotation_id": "ann-1", "image_id": "area-1", "split": "train",
             "annotation_type": "binary_qa", "task_type": "binary_qa",
             "question": "Is water present?", "provenance": {"source": "fixture"}}
    value.update(changes)
    return value


def test_text_annotation_links_image_without_inventing_spatial_geometry():
    link = link_annotation(annotation(), IMAGES, REPRESENTATIONS)
    assert link.validation_status == "valid"
    assert link.spatial_reference is None
    assert link.relationship_types == (RelationshipType.IMAGE_TO_REPRESENTATION.value,)
    assert link.representation_refs[0]["reference_id"] == "joint-1"


def test_bbox_links_tokens_with_explicit_relationships_and_fractions():
    link = link_annotation(annotation(annotation_id="box", annotation_type="text_box", task_type="text_box",
                                      box=[0.0, 0.0, 1.0, 1.0], geometry_frame="source_unit_square_unmapped"),
                          IMAGES, REPRESENTATIONS,
                          canonical_geometry=BBox(4, 4, 12, 12, CoordinateSpace.ANALYSIS_GRID))
    assert link.validation_status == "valid"
    assert RelationshipType.BBOX_TO_CROMA_TOKENS.value in link.relationship_types
    tokens = link.spatial_reference["token_links"]
    assert [item["token_index"] for item in tokens] == [0, 1, 15, 16]
    assert all(item["relationship"] == "INTERSECTS" for item in tokens)
    assert link.spatial_reference["learned_grounding"] is False


def test_point_boundary_maps_deterministically_without_clipping():
    link = link_annotation(annotation(annotation_id="point", annotation_type="point_box", task_type="point_box",
                                      point=[0.5, 0.5]), IMAGES, REPRESENTATIONS,
                          canonical_geometry=Point(8, 8, CoordinateSpace.ANALYSIS_GRID))
    assert link.validation_status == "valid"
    assert link.spatial_reference["token_index"] == 16
    outside = link_annotation(annotation(annotation_id="outside"), IMAGES, REPRESENTATIONS,
                              canonical_geometry=Point(-1, 8, CoordinateSpace.ANALYSIS_GRID))
    assert outside.validation_status == "invalid"
    assert LinkStatus.SPATIAL_LINK_UNAVAILABLE.value in outside.validation_issues


def test_source_geometry_without_canonical_space_is_unavailable():
    link = link_annotation(annotation(annotation_id="raw-box", annotation_type="text_box", task_type="text_box",
                                      box=[0.0, 0.0, 1.0, 1.0], geometry_frame="source_unit_square_unmapped"),
                          IMAGES, REPRESENTATIONS)
    assert link.validation_status == "invalid"
    assert LinkStatus.SPATIAL_LINK_UNAVAILABLE.value in link.validation_issues
    assert link.spatial_reference["source_geometry"]["box"] == [0.0, 0.0, 1.0, 1.0]


def test_split_and_provenance_mismatches_fail_closed():
    mismatch = link_annotation(annotation(split="test"), IMAGES, REPRESENTATIONS)
    assert mismatch.validation_status == "invalid"
    assert LinkStatus.SPLIT_MISMATCH.value in mismatch.validation_issues
    wrong_image = link_annotation(annotation(), IMAGES, [dict(REPRESENTATIONS[0], scene_id="area-2")])
    assert LinkStatus.PROVENANCE_MISMATCH.value in wrong_image.validation_issues
    missing_split = link_annotation(annotation(), {"area-1": {}}, REPRESENTATIONS)
    assert LinkStatus.SPLIT_UNKNOWN.value in missing_split.validation_issues


def test_missing_representation_is_explicit_and_artifact_is_deterministic(tmp_path):
    records = [annotation(annotation_id="b"), annotation(annotation_id="a", image_id="area-2", split="test")]
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_link_artifact(records, IMAGES, {"area-1": REPRESENTATIONS}, first,
                        requested_representation_types=("joint_croma", "grounding_mask"))
    write_link_artifact(records, IMAGES, {"area-1": REPRESENTATIONS}, second,
                        requested_representation_types=("joint_croma", "grounding_mask"))
    assert (first / "links.jsonl").read_bytes() == (second / "links.jsonl").read_bytes()
    assert json.loads((first / "validation_report.json").read_text())["missing_representation_cases"] == 1
    assert json.loads((first / "manifest.json").read_text())["links_sha256"] == json.loads((second / "manifest.json").read_text())["links_sha256"]
