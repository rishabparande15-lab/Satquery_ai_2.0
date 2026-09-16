from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.annotation_foundation import canonical_bytes
from src.image_language_dataset import (
    SCHEMA_VERSION,
    assign_pool,
    build_record,
    compare_deterministic_outputs,
    parse_mcq_options,
    validate_record,
    write_dataset,
)


def annotation(**changes):
    value = {
        "schema_version": "annotation_schema_v1",
        "annotation_id": "ann-1",
        "image_id": "area-1",
        "split": "train",
        "use_partition": "train",
        "task_type": "binary_qa",
        "question": "Does the satellite image display inland waters?",
        "answer": "yes",
        "options": None,
        "caption": None,
        "referenced_text": None,
        "box": None,
        "point": None,
        "geometry_frame": None,
        "coordinate_convention_status": "UNKNOWN",
        "metadata_dependency": "VISUAL_ONLY",
        "training_eligibility": "ELIGIBLE_VISUAL_VQA",
        "source_dataset": "BigEarthNet.txt",
        "source_record_id": "1",
        "source_record_sha256": "a" * 64,
        "source_task_type": "binary",
        "source_category": "presence",
        "source_partition": "train",
        "validation_status": "validated",
        "validation_issues": [],
        "provenance": {"matching_method": "dual_sentinel_exact_v1",
                       "matching_keys": ["patch_id", "s1_name"]},
    }
    value.update(changes)
    return value


def source_manifest(annotation_bytes: bytes) -> dict:
    import hashlib
    return {
        "schema_version": "annotation_schema_v1",
        "record_count": annotation_bytes.count(b"\n"),
        "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "source_revision": "pinned",
        "source_sha256": "b" * 64,
        "dataset_fingerprint": "dataset",
        "split_fingerprint": "split",
    }


def test_schema_required_nullable_and_stable_id():
    manifest = {"source_revision": "pinned", "source_sha256": "b" * 64,
                "annotations_sha256": "c" * 64}
    first = build_record(annotation(), manifest)
    second = build_record(annotation(), manifest)
    assert first == second and first["schema_version"] == SCHEMA_VERSION
    assert first["record_id"] == second["record_id"]
    assert first["options"] is first["caption"] is first["representation_ref"] is None
    broken = dict(first)
    del broken["record_id"]
    with pytest.raises(ValueError, match="Missing"):
        validate_record(broken)


def test_binary_preserves_raw_answer_and_question():
    source = annotation(question="  Is there water?  ", answer="Yes.")
    record = build_record(source, {"source_revision": "pinned"})
    assert record["question"] == "  Is there water?  "
    assert record["answer"] == record["raw_answer"] == "Yes."
    assert record["normalized_answer"] is None
    assert record["pool"] == "visual_vqa_candidate"


def test_mcq_preserves_option_order_raw_answer_and_derives_text():
    question = "Select one: a) Forest, b) Inland waters, c) Pastures, d) Urban fabric"
    options, issue = parse_mcq_options(question)
    assert issue is None
    assert [item["key"] for item in options] == ["a", "b", "c", "d"]
    record = build_record(annotation(task_type="multiple_choice_qa", source_task_type="mcq",
                                     question=question, answer="b"),
                          {"source_revision": "pinned"})
    assert record["raw_answer"] == "b" and record["normalized_answer"] == "Inland waters"
    assert [item["text"] for item in record["options"]] == [
        "Forest", "Inland waters", "Pastures", "Urban fabric"]


def test_caption_is_whole_image_and_separate_family():
    source = annotation(task_type="caption", source_task_type="captioning", question=None,
                        answer=None, caption="A whole-scene caption.")
    record = build_record(source, {"source_revision": "pinned"})
    assert record["pool"] == "captioning_candidate"
    assert record["caption"] == "A whole-scene caption."
    assert record["box"] is record["point"] is record["representation_ref"] is None


def test_metadata_policy_and_metadata_aware_pool():
    record = build_record(annotation(metadata_dependency="VISUAL_PLUS_METADATA",
                                     training_eligibility="REQUIRES_METADATA_AWARE_EXPERIMENT"),
                          {"source_revision": "pinned"})
    assert record["pool"] == "metadata_aware"
    assert record["exclusion_reason"] == "requires_separate_metadata_aware_experiment"
    metadata_only = annotation(metadata_dependency="METADATA_ONLY",
                               training_eligibility="EXCLUDED_FROM_VISUAL_VQA")
    assert assign_pool(metadata_only) == "metadata_aware"


def test_spatial_unknown_never_enters_visual_vqa():
    for task in ("text_box", "point_box"):
        source = annotation(task_type=task, metadata_dependency="SPATIAL_VISUAL",
                            training_eligibility="REQUIRES_SPATIAL_VERIFICATION",
                            question=None, answer=None, referenced_text="forest",
                            box=[0.0, 0.0, 1.0, 1.0], geometry_frame="source_unit_square_unmapped")
        record = build_record(source, {"source_revision": "pinned"})
        assert record["pool"] == "spatial_quarantine"
        assert record["coordinate_convention_status"] == "UNKNOWN"


def test_benchmark_precedence_and_invalid_unknown_quarantine():
    benchmark = annotation(use_partition="bench_holdout", validation_status="quarantined",
                           validation_issues=["benchmark_held_out"])
    assert assign_pool(benchmark) == "benchmark_holdout"
    assert assign_pool(annotation(validation_status="quarantined",
                                  validation_issues=["bad_source"])) == "unknown_quarantine"


def _write_fixture(root: Path, rows: list[dict]) -> tuple[Path, Path]:
    annotation_bytes = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    annotations = root / "annotations.jsonl"
    annotations.write_bytes(annotation_bytes)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(source_manifest(annotation_bytes)), encoding="utf-8")
    return annotations, manifest


def test_split_isolation_and_benchmark_image_isolation(tmp_path):
    rows = [annotation(annotation_id="a1", source_record_sha256="1" * 64),
            annotation(annotation_id="a2", source_record_id="2", source_record_sha256="2" * 64,
                       use_partition="bench_holdout", validation_status="quarantined",
                       validation_issues=["benchmark_held_out"])]
    annotations, manifest = _write_fixture(tmp_path, rows)
    with pytest.raises(ValueError, match="Benchmark image contamination"):
        write_dataset(annotations, manifest, tmp_path / "out")


def test_duplicate_annotations_fail_closed(tmp_path):
    rows = [annotation(), annotation()]
    annotations, manifest = _write_fixture(tmp_path, rows)
    with pytest.raises(ValueError, match="Duplicate record IDs"):
        write_dataset(annotations, manifest, tmp_path / "out")


def test_deterministic_serialization_pooling_and_audits(tmp_path):
    rows = [
        annotation(annotation_id="binary", source_record_sha256="1" * 64),
        annotation(annotation_id="mcq", source_record_id="2", source_record_sha256="2" * 64,
                   task_type="multiple_choice_qa", source_task_type="mcq",
                   question="Choose: a) Forest, b) Water", answer="b"),
        annotation(annotation_id="caption", source_record_id="3", source_record_sha256="3" * 64,
                   task_type="caption", source_task_type="captioning", question=None,
                   answer=None, caption="Scene caption."),
        annotation(annotation_id="spatial", source_record_id="4", source_record_sha256="4" * 64,
                   task_type="text_box", question=None, answer=None, referenced_text="forest",
                   metadata_dependency="SPATIAL_VISUAL",
                   training_eligibility="REQUIRES_SPATIAL_VERIFICATION"),
        annotation(annotation_id="metadata", source_record_id="5", source_record_sha256="5" * 64,
                   image_id="area-2", metadata_dependency="VISUAL_PLUS_METADATA",
                   training_eligibility="REQUIRES_METADATA_AWARE_EXPERIMENT"),
        annotation(annotation_id="bench", source_record_id="6", source_record_sha256="6" * 64,
                   image_id="area-3", split="test", use_partition="bench_holdout",
                   validation_status="quarantined", validation_issues=["benchmark_held_out"]),
    ]
    annotations, source = _write_fixture(tmp_path, rows)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = write_dataset(annotations, source, first, code_revision="head")
    write_dataset(annotations, source, second, code_revision="head")
    comparison = compare_deterministic_outputs(first, second)
    assert comparison["identical"]
    assert manifest["record_count"] == 6
    assert manifest["pools"]["visual_vqa_candidate"]["record_count"] == 2
    assert manifest["pools"]["captioning_candidate"]["record_count"] == 1
    leakage = json.loads((first / "leakage_audit.json").read_text())
    assert leakage["cross_split_image_intersections"] == {
        "train_test": 0, "train_validation": 0, "validation_test": 0}
    assert leakage["spatial_records_in_visual_vqa"] == 0
    assert json.loads((first / "provenance.json").read_text())["model_training_performed"] is False
