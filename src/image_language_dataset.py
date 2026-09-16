"""Deterministic, split-safe image-language records from verified annotations.

This module is deliberately independent of model training and Pipeline 3 runtime
code.  It only transforms the already verified annotation artifact into audited
record pools.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from src.annotation_foundation import canonical_bytes, file_digest


SCHEMA_VERSION = "image_language_record_v1"
GENERATION_VERSION = "image_language_generation_v1"
POOL_FILES = {
    "visual_vqa_candidate": "visual_vqa_candidates.jsonl",
    "metadata_aware": "metadata_aware.jsonl",
    "captioning_candidate": "caption_candidates.jsonl",
    "spatial_quarantine": "spatial_quarantine.jsonl",
    "benchmark_holdout": "benchmark_holdout.jsonl",
    "unknown_quarantine": "unknown_quarantine.jsonl",
}
TASK_TYPES = frozenset({"binary_qa", "multiple_choice_qa", "caption", "text_box", "point_box"})
DEPENDENCY_CLASSES = frozenset({
    "VISUAL_ONLY", "SPATIAL_VISUAL", "METADATA_ONLY",
    "VISUAL_PLUS_METADATA", "UNKNOWN_DEPENDENCY",
})
ELIGIBILITY_VALUES = frozenset({
    "ELIGIBLE_VISUAL_VQA", "REQUIRES_SPATIAL_VERIFICATION",
    "EXCLUDED_FROM_VISUAL_VQA", "REQUIRES_METADATA_AWARE_EXPERIMENT", "QUARANTINE",
})
_OPTION_MARKER = re.compile(r"(?<![A-Za-z0-9])([a-zA-Z])\)\s*")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _canonical_line(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed annotation JSONL line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Annotation line {line_number} is not an object")
            yield value


def parse_mcq_options(question: str | None) -> tuple[list[dict[str, str]] | None, str | None]:
    """Return source-ordered options and a failure reason without rewriting text."""
    if not isinstance(question, str):
        return None, "missing_question"
    matches = list(_OPTION_MARKER.finditer(question))
    if len(matches) < 2:
        return None, "options_not_deterministically_parseable"
    options = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        text = question[match.end():end]
        # The publisher separates options with a comma; it is a delimiter, not
        # part of the option text. Whitespace and punctuation inside each option
        # remain otherwise untouched.
        text = text.rstrip()
        if index + 1 < len(matches) and text.endswith(","):
            text = text[:-1].rstrip()
        options.append({"key": match.group(1), "text": text})
    keys = [option["key"].lower() for option in options]
    if len(keys) != len(set(keys)):
        return None, "duplicate_option_keys"
    return options, None


def _exclusion_reason(annotation: dict[str, Any], pool: str) -> str | None:
    if pool == "visual_vqa_candidate":
        return None
    if pool == "benchmark_holdout":
        return "benchmark_image_isolated_from_training_and_model_selection"
    if pool == "spatial_quarantine":
        return "coordinate_convention_and_raster_mapping_are_unknown"
    if pool == "metadata_aware":
        if annotation.get("metadata_dependency") == "METADATA_ONLY":
            return "answer_depends_on_non_visual_metadata"
        return "requires_separate_metadata_aware_experiment"
    if annotation.get("validation_issues"):
        return ";".join(annotation["validation_issues"])
    return "unknown_dependency_or_unsupported_record"


def assign_pool(annotation: dict[str, Any]) -> str:
    """Assign exactly one conservative pool; benchmark isolation has precedence."""
    if annotation.get("use_partition") == "bench_holdout":
        return "benchmark_holdout"
    if annotation.get("validation_status") != "validated":
        return "unknown_quarantine"
    task = annotation.get("task_type")
    eligibility = annotation.get("training_eligibility")
    dependency = annotation.get("metadata_dependency")
    if task in {"text_box", "point_box"}:
        return "spatial_quarantine"
    if task == "caption":
        return "captioning_candidate"
    if task in {"binary_qa", "multiple_choice_qa"}:
        if eligibility == "ELIGIBLE_VISUAL_VQA" and dependency == "VISUAL_ONLY":
            return "visual_vqa_candidate"
        if dependency in {"METADATA_ONLY", "VISUAL_PLUS_METADATA"}:
            return "metadata_aware"
    return "unknown_quarantine"


def build_record(annotation: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Construct one canonical record while retaining source text and identity."""
    annotation_id = annotation.get("annotation_id")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError("annotation_id is required")
    task = annotation.get("task_type")
    pool = assign_pool(annotation)
    parsed_options = None
    option_parse_status = None
    normalized_answer = None
    if task == "multiple_choice_qa":
        parsed_options, option_parse_status = parse_mcq_options(annotation.get("question"))
        if parsed_options is not None and isinstance(annotation.get("answer"), str):
            answer_key = annotation["answer"].strip().lower()
            matches = [option["text"] for option in parsed_options if option["key"].lower() == answer_key]
            if len(matches) == 1:
                normalized_answer = matches[0]
            else:
                option_parse_status = "answer_key_not_in_options"
    record_id = f"ilr:{_sha256_bytes(canonical_bytes([SCHEMA_VERSION, annotation_id]))}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "annotation_id": annotation_id,
        "image_id": annotation.get("image_id"),
        "split": annotation.get("split"),
        "effective_partition": annotation.get("use_partition"),
        "pool": pool,
        "task_type": task,
        "question": annotation.get("question"),
        "answer": annotation.get("answer"),
        "raw_answer": annotation.get("answer"),
        "normalized_answer": normalized_answer,
        "options": parsed_options if task == "multiple_choice_qa" else annotation.get("options"),
        "raw_options": annotation.get("options"),
        "option_parse_status": option_parse_status,
        "caption": annotation.get("caption"),
        "referenced_text": annotation.get("referenced_text"),
        "box": annotation.get("box"),
        "point": annotation.get("point"),
        "geometry_frame": annotation.get("geometry_frame"),
        "coordinate_convention_status": annotation.get("coordinate_convention_status"),
        "dependency_class": annotation.get("metadata_dependency"),
        "eligibility": annotation.get("training_eligibility"),
        "exclusion_reason": _exclusion_reason(annotation, pool),
        "source_dataset": annotation.get("source_dataset"),
        "source_revision": source_manifest.get("source_revision"),
        "source_record_id": annotation.get("source_record_id"),
        "source_record_sha256": annotation.get("source_record_sha256"),
        "source_task_type": annotation.get("source_task_type"),
        "source_category": annotation.get("source_category"),
        "source_partition": annotation.get("source_partition"),
        "provenance": {
            "annotation_schema_version": annotation.get("schema_version"),
            "annotation_artifact_sha256": source_manifest.get("annotations_sha256"),
            "source_sha256": source_manifest.get("source_sha256"),
            "matching_method": annotation.get("provenance", {}).get("matching_method"),
            "matching_keys": annotation.get("provenance", {}).get("matching_keys"),
        },
        # Reserved for Phase 6B.5+; never populated speculatively here.
        "representation_ref": None,
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version", "record_id", "annotation_id", "image_id", "split", "pool",
        "task_type", "question", "answer", "options", "caption", "source_dataset",
        "source_revision", "provenance", "eligibility", "dependency_class", "exclusion_reason",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Missing image-language fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported image-language schema")
    if record["task_type"] not in TASK_TYPES:
        raise ValueError("Unsupported task type")
    if record["dependency_class"] not in DEPENDENCY_CLASSES:
        raise ValueError("Unsupported dependency class")
    if record["eligibility"] not in ELIGIBILITY_VALUES:
        raise ValueError("Unsupported eligibility")
    if record["pool"] not in POOL_FILES:
        raise ValueError("Unsupported pool")
    if not isinstance(record["record_id"], str) or not isinstance(record["annotation_id"], str):
        raise ValueError("Record and annotation IDs must be strings")
    if not isinstance(record["image_id"], str) or record["split"] not in {"train", "validation", "test"}:
        raise ValueError("Canonical image identity and split are required")
    if record["task_type"] in {"binary_qa", "multiple_choice_qa"} and (
        not isinstance(record["question"], str) or not isinstance(record["answer"], str)
    ):
        raise ValueError("Q&A records require original question and answer strings")
    if record["task_type"] == "caption" and not isinstance(record["caption"], str):
        raise ValueError("Caption records require the original caption")
    if record["pool"] == "visual_vqa_candidate" and (
        record["task_type"] not in {"binary_qa", "multiple_choice_qa"}
        or record["dependency_class"] != "VISUAL_ONLY"
        or record["eligibility"] != "ELIGIBLE_VISUAL_VQA"
    ):
        raise ValueError("Visual VQA pool contains an ineligible record")


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _pool_summary(records: list[dict[str, Any]], checksum: str) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "image_count": len({record["image_id"] for record in records}),
        "split_counts": _counter(record["split"] for record in records),
        "partition_counts": _counter(record["effective_partition"] for record in records),
        "task_counts": _counter(record["task_type"] for record in records),
        "dependency_class_counts": _counter(record["dependency_class"] for record in records),
        "eligibility_counts": _counter(record["eligibility"] for record in records),
        "artifact_sha256": checksum,
    }


def _duplicate_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    def excess(values: Iterable[Any]) -> int:
        return sum(count - 1 for count in Counter(values).values() if count > 1)

    qa = [record for record in records if record["task_type"] in {"binary_qa", "multiple_choice_qa"}]
    pair_images: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    question_images: dict[str, set[str]] = defaultdict(set)
    for record in qa:
        pair = (record["question"], record["answer"])
        pair_images[pair].add(record["image_id"])
        pair_splits[pair].add(record["split"])
        question_images[record["question"]].add(record["image_id"])
    split_images = {name: {r["image_id"] for r in records if r["split"] == name}
                    for name in ("train", "validation", "test")}
    cross_split_pairs = {pair: splits for pair, splits in pair_splits.items() if len(splits) > 1}
    repeated_pairs = {pair: images for pair, images in pair_images.items() if len(images) > 1}
    image_counts = Counter(record["image_id"] for record in records)
    return {
        "unique_images": len(image_counts),
        "images_with_multiple_annotation_records": sum(count > 1 for count in image_counts.values()),
        "repeated_image_references": sum(count - 1 for count in image_counts.values()),
        "repeated_image_references_are_expected_annotations_on_the_same_image": True,
        "duplicate_record_ids": excess(record["record_id"] for record in records),
        "duplicate_annotation_ids": excess(record["annotation_id"] for record in records),
        "duplicate_source_records": excess(record["source_record_sha256"] for record in records),
        "duplicate_image_question": excess((r["image_id"], r["question"]) for r in qa),
        "duplicate_image_question_answer": excess((r["image_id"], r["question"], r["answer"]) for r in qa),
        "identical_question_answer_across_different_images": len(repeated_pairs),
        "identical_question_across_different_images": sum(len(images) > 1 for images in question_images.values()),
        "cross_split_question_answer_overlap": len(cross_split_pairs),
        "cross_split_image_intersections": {
            "train_validation": len(split_images["train"] & split_images["validation"]),
            "train_test": len(split_images["train"] & split_images["test"]),
            "validation_test": len(split_images["validation"] & split_images["test"]),
        },
        "repeated_content_is_not_automatically_image_leakage": True,
    }


def _eligibility_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    qa = [r for r in records if r["task_type"] in {"binary_qa", "multiple_choice_qa"}]
    binary = [r for r in qa if r["task_type"] == "binary_qa"]
    mcq = [r for r in qa if r["task_type"] == "multiple_choice_qa"]
    questions = Counter(r["question"] for r in qa)
    disposition_by_task = {}
    for task in sorted(TASK_TYPES):
        task_records = [record for record in records if record["task_type"] == task]
        disposition_by_task[task] = {
            "source_records": len(task_records),
            "candidate_records": sum(record["pool"] in {
                "visual_vqa_candidate", "metadata_aware", "captioning_candidate"
            } for record in task_records),
            "eligible_visual_vqa_records": sum(
                record["pool"] == "visual_vqa_candidate" for record in task_records),
            "excluded_or_restricted_records": sum(
                record["pool"] == "metadata_aware" for record in task_records),
            "quarantined_or_held_out_records": sum(record["pool"] in {
                "spatial_quarantine", "benchmark_holdout", "unknown_quarantine"
            } for record in task_records),
        }
    return {
        "source_records": len(records),
        "candidate_records": sum(r["pool"] in {"visual_vqa_candidate", "metadata_aware"} for r in records),
        "eligible_records": sum(r["pool"] == "visual_vqa_candidate" for r in records),
        "excluded_records": sum(r["pool"] == "metadata_aware" for r in records),
        "quarantined_records": sum(r["pool"] in {"spatial_quarantine", "benchmark_holdout", "unknown_quarantine"} for r in records),
        "counts_by_task": _counter(r["task_type"] for r in records),
        "counts_by_split": _counter(r["split"] for r in records),
        "counts_by_dependency_class": _counter(r["dependency_class"] for r in records),
        "counts_by_eligibility": _counter(r["eligibility"] for r in records),
        "pool_counts": _counter(r["pool"] for r in records),
        "record_disposition_by_task": disposition_by_task,
        "task_split_dependency_counts": {
            f"{task}|{split}|{dependency}": count for (task, split, dependency), count in sorted(Counter(
                (r["task_type"], r["split"], r["dependency_class"]) for r in records
            ).items())
        },
        "binary_answer_distribution": _counter(r["raw_answer"] for r in binary),
        "mcq_answer_key_distribution": _counter(r["raw_answer"] for r in mcq),
        "mcq_option_count_distribution": _counter(len(r["options"]) if r["options"] else 0 for r in mcq),
        "mcq_option_parse_status": _counter(r["option_parse_status"] or "parsed" for r in mcq),
        "metadata_risk_category_counts": _counter(
            r["source_category"] for r in records if r["pool"] == "metadata_aware"),
        "visual_vqa_metadata_risk_category_counts": _counter(
            r["source_category"] for r in records
            if r["pool"] == "visual_vqa_candidate"
            and r["source_category"] in {"country", "season", "climate zone", "climate_zone"}),
        "question_frequency_top_50": [
            {"question": question, "count": count}
            for question, count in sorted(questions.items(), key=lambda item: (-item[1], item[0]))[:50]
        ],
        "question_frequency_histogram": _counter(questions.values()),
        "descriptive_only_no_resampling": True,
    }


def _assert_isolation(records: list[dict[str, Any]]) -> None:
    by_image: dict[str, set[str]] = defaultdict(set)
    benchmark_images = set()
    for record in records:
        by_image[record["image_id"]].add(record["split"])
        if record["pool"] == "benchmark_holdout":
            benchmark_images.add(record["image_id"])
    if any(len(splits) != 1 for splits in by_image.values()):
        raise ValueError("Cross-split image leakage detected")
    if any(r["image_id"] in benchmark_images and r["pool"] != "benchmark_holdout" for r in records):
        raise ValueError("Benchmark image contamination detected")
    visual = [r for r in records if r["pool"] == "visual_vqa_candidate"]
    if any(r["task_type"] in {"text_box", "point_box"} for r in visual):
        raise ValueError("Spatial record entered visual VQA pool")
    if any(r["dependency_class"] != "VISUAL_ONLY" for r in visual):
        raise ValueError("Metadata-dependent record entered visual VQA pool")


def write_dataset(annotation_path: Path, source_manifest_path: Path, output: Path, *,
                  code_revision: str | None = None) -> dict[str, Any]:
    """Write deterministic pools and canonical audits; timestamp only provenance."""
    annotation_path, source_manifest_path, output = map(Path, (annotation_path, source_manifest_path, output))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if file_digest(annotation_path) != source_manifest.get("annotations_sha256"):
        raise ValueError("Verified annotation artifact checksum mismatch")
    records = [build_record(annotation, source_manifest) for annotation in read_jsonl(annotation_path)]
    records.sort(key=lambda record: record["record_id"])
    _assert_isolation(records)
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("Duplicate record IDs detected")
    if len({record["annotation_id"] for record in records}) != len(records):
        raise ValueError("Duplicate annotation IDs detected")

    output.mkdir(parents=True, exist_ok=True)
    pools = {pool: [] for pool in POOL_FILES}
    for record in records:
        pools[record["pool"]].append(record)
    checksums = {}
    for pool, filename in POOL_FILES.items():
        path = output / filename
        with path.open("wb") as stream:
            for record in pools[pool]:
                stream.write(_canonical_line(record))
        checksums[pool] = file_digest(path)

    leakage = _duplicate_audit(records)
    leakage.update({
        "benchmark_records_outside_holdout": sum(
            r["effective_partition"] == "bench_holdout" and r["pool"] != "benchmark_holdout" for r in records),
        "spatial_records_in_visual_vqa": sum(
            r["pool"] == "visual_vqa_candidate" and r["task_type"] in {"text_box", "point_box"} for r in records),
        "metadata_dependent_records_in_visual_vqa": sum(
            r["pool"] == "visual_vqa_candidate" and r["dependency_class"] != "VISUAL_ONLY" for r in records),
    })
    eligibility = _eligibility_audit(records)
    canonical_record_fingerprint = _sha256_bytes(b"".join(_canonical_line(record) for record in records))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generation_version": GENERATION_VERSION,
        "source_dataset": "BigEarthNet.txt",
        "source_revision": source_manifest.get("source_revision"),
        "source_sha256": source_manifest.get("source_sha256"),
        "source_annotation_schema": source_manifest.get("schema_version"),
        "source_annotation_count": source_manifest.get("record_count"),
        "source_annotations_sha256": source_manifest.get("annotations_sha256"),
        "dataset_fingerprint": source_manifest.get("dataset_fingerprint"),
        "split_fingerprint": source_manifest.get("split_fingerprint"),
        "image_language_fingerprint": canonical_record_fingerprint,
        "record_count": len(records),
        "pools": {
            pool: {
                "file": POOL_FILES[pool],
                "schema_version": SCHEMA_VERSION,
                "generation_version": GENERATION_VERSION,
                "source_revision": source_manifest.get("source_revision"),
                "source_sha256": source_manifest.get("source_sha256"),
                "dataset_fingerprint": source_manifest.get("dataset_fingerprint"),
                "split_fingerprint": source_manifest.get("split_fingerprint"),
                **_pool_summary(values, checksums[pool]),
            }
            for pool, values in pools.items()
        },
    }
    reports = {
        "manifest.json": manifest,
        "eligibility_audit.json": {
            "schema_version": SCHEMA_VERSION,
            "source_revision": source_manifest.get("source_revision"),
            **eligibility,
        },
        "leakage_audit.json": {
            "schema_version": SCHEMA_VERSION,
            "source_revision": source_manifest.get("source_revision"),
            **leakage,
        },
    }
    for filename, value in reports.items():
        (output / filename).write_bytes(_canonical_line(value))
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generation_version": GENERATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "source_annotation_file": annotation_path.name,
        "source_manifest_file": source_manifest_path.name,
        "canonical_files": {
            **{POOL_FILES[pool]: checksums[pool] for pool in POOL_FILES},
            **{filename: file_digest(output / filename) for filename in reports},
        },
        "timestamp_excluded_from_canonical_hashes": True,
        "pipeline3_modified": False,
        "model_training_performed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def compare_deterministic_outputs(first: Path, second: Path) -> dict[str, Any]:
    files = sorted({*POOL_FILES.values(), "manifest.json", "eligibility_audit.json", "leakage_audit.json"})
    comparisons = {name: file_digest(Path(first) / name) == file_digest(Path(second) / name) for name in files}
    return {"identical": all(comparisons.values()), "files": comparisons}
