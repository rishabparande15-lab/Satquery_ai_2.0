from __future__ import annotations

import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.architecture_contracts import ArtifactRef, ArtifactType
from src.annotation_foundation import (SCHEMA_VERSION, ingest, load_images, parse_geometry, read_source,
                                       write_artifact)


def fixtures(tmp_path):
    images = {"format_version": 1, "rows": [
        {"area_id": "area-a", "s2_identity": "s2-a", "s1_identity": "s1-a", "split": "train"},
        {"area_id": "area-b", "s2_identity": "s2-b", "s1_identity": "s1-b", "split": "test"}]}
    splits = {"format_version": 1, "ids": {"train": ["area-a"], "validation": [], "test": ["area-b"]}}
    manifest, split = tmp_path / "images.json", tmp_path / "split.json"
    manifest.write_text(json.dumps(images), encoding="utf-8")
    split.write_text(json.dumps(splits), encoding="utf-8")
    return manifest, split


def source_row(**overrides):
    row = {"ID": "q1", "patch_id": "s2-a", "s1_name": "s1-a", "type": "binary",
           "input": "  Is there water?  ", "output": "Yes.", "split": "test"}
    row.update(overrides)
    return row


def test_exact_dual_sensor_match_split_inheritance_and_raw_text(tmp_path):
    index, _ = load_images(*fixtures(tmp_path))
    records, split, leakage = ingest([source_row(), source_row(ID="q2", s1_name="wrong")], index)
    valid = next(r for r in records if r["source_record_id"] == "q1")
    assert valid["schema_version"] == SCHEMA_VERSION
    assert valid["image_id"] == "area-a" and valid["split"] == "train"
    assert valid["source_partition"] == "test" and valid["validation_status"] == "quarantined"
    assert "source_partition_conflict" in valid["validation_issues"]
    assert valid["question"] == "  Is there water?  " and valid["raw_source"]["output"] == "Yes."
    assert valid["box"] is None and valid["caption"] is None
    assert valid["provenance"]["matching_keys"] == ["patch_id", "s1_name"]
    assert next(r for r in records if r["source_record_id"] == "q2")["validation_status"] == "quarantined"
    assert split["source_partition_disagreements"] == 1
    assert leakage["image_level_leakage"] == 0


def test_duplicate_and_cross_split_qa_detection(tmp_path):
    index, _ = load_images(*fixtures(tmp_path))
    rows = [source_row(), source_row(), source_row(ID="q3", patch_id="s2-b", s1_name="s1-b")]
    records, _, leakage = ingest(rows, index)
    assert leakage["duplicate_annotation_ids"] == 1
    assert leakage["duplicate_source_records"] == 1
    assert leakage["identical_qa_across_splits"] == 1
    assert sum(r["validation_status"] == "quarantined" for r in records) == 2
    assert leakage["identical_qa_across_splits_is_not_image_leakage"]


def test_source_ids_task_validation_and_geometry(tmp_path):
    index, _ = load_images(*fixtures(tmp_path))
    rows = [source_row(ID="same", type="bogus"), source_row(ID="same", input="Different"),
            source_row(ID="box", type="bounding box", category="reference", box=[3, 2, 1, 4]),
            source_row(ID="caption", type="captioning", output=None),
            source_row(ID="held", split="bench")]
    records, _, leakage = ingest(rows, index)
    assert leakage["duplicate_source_ids_with_different_content"] == 1
    issues = {r["source_record_id"]: r["validation_issues"] for r in records}
    assert "unknown_task_type" in issues["same"] or "duplicate_source_id_conflict" in issues["same"]
    assert "malformed_source_box" in issues["box"]
    assert "missing_caption" in issues["caption"]
    assert "benchmark_held_out" in issues["held"]


def test_live_box_and_point_string_geometry_with_quarantine(tmp_path):
    index, _ = load_images(*fixtures(tmp_path))
    good = source_row(ID="p", type="bounding box", category="point", split="train",
                      input="Locate <point>(0.82, 0.28)</point>.", output="[0.64 0.0, 1.0 0.71]")
    bad = source_row(ID="bad", type="bounding box", category="point", split="train",
                     input="Locate <point>(1.5, 0.28)</point>.", output="[0.64 0.0, 1.0 0.71]")
    malformed = source_row(ID="malformed", type="bounding box", category="reference", split="train",
                           input="Locate <ref>pastures</ref>.", output="[bad]")
    records, _, _ = ingest([good, bad, malformed], index)
    by_id = {r["source_record_id"]: r for r in records}
    assert by_id["p"]["box"] == [0.64, 0.0, 1.0, 0.71]
    assert by_id["p"]["point"] == [0.82, 0.28]
    assert by_id["p"]["raw_source"]["output"] == "[0.64 0.0, 1.0 0.71]"
    assert by_id["p"]["geometry_frame"] == "source_unit_square_unmapped"
    assert by_id["p"]["validation_status"] == "validated"
    assert "out_of_unit_square_point" in by_id["bad"]["validation_issues"]
    assert "malformed_source_box" in by_id["malformed"]["validation_issues"]
    assert by_id["malformed"]["referenced_text"] == "pastures"
    assert parse_geometry(good)[2] == "source_unit_square_unmapped"
    repeated = source_row(ID="repeat", type="bounding box", category="point", split="train",
                          input="<point>(0.82, 0.28)</point> and <point>(0.82, 0.28)</point>",
                          output="[0.64 0.0, 1.0 0.71]")
    assert parse_geometry(repeated)[1] == [0.82, 0.28]
    overflow = source_row(type="bounding box", category="point", split="train",
                          input="<point>(1e999, 0.2)</point>", output="[0.0 0.0, 1.0 1.0]")
    parsed_box, parsed_point, _, overflow_issues = parse_geometry(overflow)
    assert parsed_box == [0.0, 0.0, 1.0, 1.0] and parsed_point is None
    assert "nonfinite_source_point" in overflow_issues


def test_benchmark_holds_out_every_annotation_on_image(tmp_path):
    index, _ = load_images(*fixtures(tmp_path))
    rows = [source_row(ID="bench", split="bench"), source_row(ID="train", split="train")]
    records, audit, leakage = ingest(rows, index)
    assert all(r["use_partition"] == "bench_holdout" for r in records)
    assert all("benchmark_held_out" in r["validation_issues"] for r in records)
    assert audit["use_partition_annotation_counts"]["bench_holdout"] == 2
    assert leakage["benchmark_held_out_records"] == 2


def test_partial_and_conflicting_sensor_identities(tmp_path):
    index, _ = load_images(*fixtures(tmp_path))
    rows = [source_row(ID="partial", s1_name="absent"),
            source_row(ID="conflict", s1_name="s1-b")]
    records, _, leakage = ingest(rows, index)
    assert {r["match_status"] for r in records} == {"partial_match", "conflicting_identity"}
    assert leakage["match_categories"] == {"conflicting_identity": 1, "partial_match": 1}


def test_manifest_split_conflict_and_malformed_source(tmp_path):
    manifest, split = fixtures(tmp_path)
    payload = json.loads(split.read_text())
    payload["ids"]["validation"].append("area-a")
    split.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="across splits"):
        load_images(manifest, split)
    source = tmp_path / "source.jsonl"
    source.write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        list(read_source(source))


def test_deterministic_semantic_serialization_and_provenance(tmp_path):
    manifest, split = fixtures(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(source_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "artifact"
    first = write_artifact(source, manifest, split, output, source_revision="pinned")
    first_bytes = (output / "annotations.jsonl").read_bytes()
    provenance = json.loads((output / "provenance.json").read_text())
    assert first["source_records"] == first["matched_records"] == 1
    assert first["scanned_source_records"] == 1
    assert provenance["source_revision"] == "pinned"
    assert provenance["source_sha256"] and provenance["split_manifest_sha256"]
    assert provenance["matching_implementation_sha256"]
    assert "\\" not in json.dumps(provenance)
    write_artifact(source, manifest, split, output, source_revision="pinned")
    assert first_bytes == (output / "annotations.jsonl").read_bytes()
    assert json.loads(first_bytes)["raw_source"] == source_row()
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_artifact(source, manifest, split, output, expected_source_sha256="0" * 64)


def test_parquet_scan_scope_and_architecture_reference(tmp_path):
    manifest, split = fixtures(tmp_path)
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        source_row(split="train"),
        source_row(ID="outside", patch_id="other", s1_name="other", split="train"),
        source_row(ID="wrong-sar", s1_name="wrong", split="train")]), source)
    output = tmp_path / "artifact"
    report = write_artifact(source, manifest, split, output)
    assert report["scanned_source_records"] == 3
    assert report["outside_selected_image_inventory"] == 1
    assert report["source_records"] == 2 and report["unmatched_records"] == 1
    schema = json.loads((output / "source_schema.json").read_text())
    assert schema["declared_rows"] == schema["scanned_rows"] == 3
    assert schema["columns"] and schema["type_counts"] == {"binary": 3}
    for name in ("manifest.json", "source_schema.json", "validation_report.json",
                 "split_audit.json", "leakage_audit.json", "geometry_audit.json"):
        payload = json.loads((output / name).read_text())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["source_sha256"] and payload["matching_implementation_sha256"]
    receipt = json.loads((output / "manifest.json").read_text())
    reference = ArtifactRef(artifact_id="annotations-v1", artifact_type=ArtifactType.JSON_DOCUMENT,
                            uri="artifact://annotations/bigearthnet-txt/annotations.jsonl",
                            checksum_sha256=receipt["annotations_sha256"],
                            media_type="application/jsonl", format="jsonl")
    assert reference.to_dict()["checksum_sha256"] == receipt["annotations_sha256"]


def test_pinned_official_sample_against_frozen_manifest():
    root = Path(__file__).resolve().parents[1]
    source = root / "tests/fixtures/bigearthnet_txt_official_sample.jsonl"
    index, _ = load_images(root / "experiments/pipeline3_5000/dataset_manifest.json",
                           root / "experiments/pipeline3_5000/split_manifest.json")
    raw = list(read_source(source))
    records, audit, leakage = ingest(raw, index, source_revision="72d865f2146f0a85b720f7f3ca1cdbaeafc3d316")
    assert len(records) == 5 and all(r["match_status"] == "exact_match" for r in records)
    assert all(r["split"] == r["use_partition"] == "test" for r in records)
    assert {r["task_type"] for r in records} == {"binary_qa", "multiple_choice_qa", "caption", "text_box", "point_box"}
    assert all(r["validation_status"] == "validated" for r in records)
    assert all(r["raw_source"] in raw for r in records)
    assert audit["matched_image_counts"] == {"test": 1}
    assert leakage["image_level_leakage"] == 0


def test_coordinate_convention_and_metadata_policy_are_explicitly_unknown_or_conservative():
    from src.annotation_foundation import (
        COORDINATE_CONVENTION_STATUS,
        classify_metadata_dependency,
        training_eligibility_for,
    )

    assert COORDINATE_CONVENTION_STATUS == "UNKNOWN"
    assert classify_metadata_dependency("caption", "Describe the scene") == "VISUAL_ONLY"
    assert classify_metadata_dependency("multiple_choice_qa", "What country is shown?") == "METADATA_ONLY"
    assert classify_metadata_dependency("multiple_choice_qa", "Which object is in the image?", answer="tree") == "VISUAL_ONLY"
    assert classify_metadata_dependency("text_box", "Find the building") == "SPATIAL_VISUAL"
    assert classify_metadata_dependency("binary_qa", "Does the scene show a lake?") == "VISUAL_ONLY"
    assert training_eligibility_for("METADATA_ONLY") == "EXCLUDED_FROM_VISUAL_VQA"
    assert training_eligibility_for("VISUAL_ONLY") == "ELIGIBLE_VISUAL_VQA"
    assert training_eligibility_for("UNKNOWN_DEPENDENCY") == "QUARANTINE"
