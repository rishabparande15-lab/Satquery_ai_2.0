"""Versioned, conservative BigEarthNet.txt linkage; independent of Pipeline 3."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "annotation_schema_v1"
MATCH_VERSION = "dual_sentinel_exact_v1"
OFFICIAL_REVISION = "72d865f2146f0a85b720f7f3ca1cdbaeafc3d316"
OFFICIAL_SHA256 = "d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554"
OFFICIAL_URL = ("https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/"
                f"{OFFICIAL_REVISION}/BigEarthNet.txt.parquet")
TASKS = {"binary": "binary_qa", "mcq": "multiple_choice_qa",
         "captioning": "caption", "bounding box": "text_box", "region": "point_box"}
SPLITS = {"train", "validation", "test"}
COORDINATE_CONVENTION_STATUS = "UNKNOWN"
METADATA_ONLY_KEYWORDS = ("country", "season", "climate", "climate_zone", "climate zone")
VISUAL_KEYWORDS = ("object", "building", "water", "tree", "road", "lake", "forest", "sky",
                  "person", "animal", "vehicle", "scene", "image", "field", "village")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_BOX = re.compile(rf"^\[\s*({_NUMBER})\s+({_NUMBER})\s*,\s*({_NUMBER})\s+({_NUMBER})\s*\]$")
_POINT = re.compile(rf"<point>\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)</point>")
_REF = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)


def classify_metadata_dependency(task_type: str | None, question: str | None = None,
                                 answer: str | None = None, source_category: str | None = None) -> str:
    """Classify a task by dependency on metadata versus visual/spatial evidence.

    The coordinate convention itself remains UNKNOWN, so this policy is deliberately
    conservative and excludes any task that depends on source metadata from visual-only
    VQA training until a separate metadata-aware experiment is designed.
    """
    task_type = (task_type or "").strip().lower()
    if task_type in {"caption", "captioning"}:
        return "VISUAL_ONLY"
    if task_type in {"text_box", "point_box"}:
        return "SPATIAL_VISUAL"
    if task_type not in {"binary_qa", "multiple_choice_qa"}:
        return "UNKNOWN_DEPENDENCY"
    context = " ".join(part for part in (question or "", answer or "", source_category or "") if part).lower()
    has_metadata = any(keyword in context for keyword in METADATA_ONLY_KEYWORDS)
    if not has_metadata:
        return "VISUAL_ONLY"
    visual_hint = any(keyword in context for keyword in VISUAL_KEYWORDS)
    if visual_hint:
        return "VISUAL_PLUS_METADATA"
    return "METADATA_ONLY"


def training_eligibility_for(classification: str | None) -> str:
    """Map the metadata dependency taxonomy to a conservative training policy."""
    mapping = {
        "VISUAL_ONLY": "ELIGIBLE_VISUAL_VQA",
        "SPATIAL_VISUAL": "REQUIRES_SPATIAL_VERIFICATION",
        "METADATA_ONLY": "EXCLUDED_FROM_VISUAL_VQA",
        "VISUAL_PLUS_METADATA": "REQUIRES_METADATA_AWARE_EXPERIMENT",
        "UNKNOWN_DEPENDENCY": "QUARANTINE",
    }
    return mapping.get((classification or "").upper(), "QUARANTINE")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_source(path: Path) -> Iterable[dict[str, Any]]:
    """Read source rows without assigning identity from row order."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL source line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Source line {line_number} is not an object")
                yield row
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()
    else:
        raise ValueError("Source must be JSONL or Parquet")


def load_images(manifest_path: Path, split_path: Path) -> tuple[dict[tuple[str, str], dict], dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or split.get("format_version") != 1:
        raise ValueError("Unsupported image/split manifest")
    ids = split.get("ids", {})
    if set(ids) != SPLITS or any(len(v) != len(set(v)) for v in ids.values()):
        raise ValueError("Invalid or duplicate split identifiers")
    owner = {}
    for name, values in ids.items():
        for image_id in values:
            if image_id in owner:
                raise ValueError("Image identifier occurs across splits")
            owner[image_id] = name
    index, seen_s1, seen_s2 = {}, {}, {}
    for image in manifest["rows"]:
        area = image["area_id"]
        if area not in owner or image.get("split") != owner[area]:
            raise ValueError("Image manifest disagrees with split manifest")
        s2, s1 = image["s2_identity"], image["s1_identity"]
        for identity, seen in ((s1, seen_s1), (s2, seen_s2)):
            if identity in seen and seen[identity] != area:
                raise ValueError("Sensor identity shared by multiple areas")
            seen[identity] = area
        for key in ((s2, s1), (area, s1)):
            if key in index:
                raise ValueError("Duplicate image association key")
            index[key] = image
    if len(seen_s1) != len(owner) or len(seen_s2) != len(owner):
        raise ValueError("Image and split inventories differ")
    fingerprint_path = Path(manifest_path).parent / "fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")) if fingerprint_path.exists() else None
    if fingerprint and fingerprint["basis"]["split_sha256"] != split.get("split_sha256"):
        raise ValueError("Frozen dataset fingerprint disagrees with split manifest")
    return index, {"image_count": len(owner), "image_manifest_sha256": file_digest(manifest_path),
                   "split_manifest_sha256": file_digest(split_path),
                   "dataset_fingerprint": fingerprint["fingerprint"] if fingerprint else None,
                   "split_fingerprint": split.get("split_sha256")}


def _source_id(row: dict) -> str | None:
    value = row.get("ID", row.get("id"))
    return str(value) if value is not None and str(value).strip() else None


def _validate_record(record: dict) -> list[str]:
    issues = []
    if record["task_type"] not in set(TASKS.values()):
        issues.append("unknown_task_type")
    if not isinstance(record["raw_source"], dict):
        issues.append("invalid_raw_source")
    if not record["optical_identity"] or not record["sar_identity"]:
        issues.append("missing_sensor_identity")
    if record["task_type"] in {"binary_qa", "multiple_choice_qa"} and (
        not isinstance(record["question"], str) or not record["question"].strip()
        or not isinstance(record["answer"], str) or not record["answer"].strip()
    ):
        issues.append("missing_question_or_answer")
    if record["task_type"] == "caption" and (
        not isinstance(record["caption"], str) or not record["caption"].strip()
    ):
        issues.append("missing_caption")
    if record["task_type"] == "text_box" and (
        not isinstance(record["referenced_text"], str) or not record["referenced_text"].strip()
    ):
        issues.append("missing_box_text")
    for field, count in (("box", 4), ("point", 2)):
        value = record[field]
        if value is not None and (not isinstance(value, (list, tuple)) or len(value) != count
                                  or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                         or not math.isfinite(v) for v in value)):
            issues.append(f"invalid_{field}_coordinates")
        elif field == "box" and value is not None and (value[2] <= value[0] or value[3] <= value[1]):
            issues.append("invalid_box_coordinates")
    return issues


def parse_geometry(row: dict) -> tuple[list[float] | None, list[float] | None, str | None, list[str]]:
    """Parse publisher's observed unit-square strings; keep source strings untouched."""
    if row.get("type") != "bounding box":
        return row.get("box"), row.get("point"), None, []
    issues = []
    output = row.get("output")
    match = _BOX.fullmatch(output) if isinstance(output, str) else None
    box = [float(v) for v in match.groups()] if match else None
    if box is None:
        issues.append("malformed_source_box")
    elif not all(math.isfinite(v) for v in box):
        issues.append("nonfinite_source_box")
        box = None
    elif not all(0 <= v <= 1 for v in box):
        issues.append("out_of_unit_square_box")
    elif box[2] <= box[0] or box[3] <= box[1]:
        issues.append("degenerate_source_box")
    point = None
    if row.get("category") == "point":
        instruction = row.get("input")
        matches = _POINT.findall(instruction) if isinstance(instruction, str) else []
        if not matches or len(set(matches)) != 1:
            issues.append("malformed_or_ambiguous_source_point")
        else:
            point = [float(v) for v in matches[0]]
            if not all(math.isfinite(v) for v in point):
                issues.append("nonfinite_source_point")
                point = None
            elif not all(0 <= v <= 1 for v in point):
                issues.append("out_of_unit_square_point")
            elif box and not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
                issues.append("source_point_outside_box")
    return box, point, "source_unit_square_unmapped", issues


def ingest(rows: Iterable[dict], image_index: dict[tuple[str, str], dict], *,
           source_revision: str | None = None) -> tuple[list[dict], dict, dict]:
    records = []
    source_ids, qa_splits = defaultdict(set), defaultdict(set)
    image_partitions = defaultdict(set)
    image_questions = Counter()
    optical_owners = {optical: image for (optical, _), image in image_index.items()}
    sar_owners = {sar: image for (_, sar), image in image_index.items()}
    audit = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Source row is not an object")
        raw = dict(row)
        optical, sar = row.get("patch_id"), row.get("s1_name")
        if not isinstance(optical, str): optical = None
        if not isinstance(sar, str): sar = None
        image = image_index.get((optical, sar)) if optical and sar else None
        if image:
            match_status = "exact_match"
        elif optical and optical in optical_owners and sar and sar in sar_owners:
            match_status = "conflicting_identity"
        elif optical and optical in optical_owners:
            match_status = "partial_match"
        else:
            match_status = "unmatched"
        source_type = row.get("type")
        task = TASKS.get(source_type) if isinstance(source_type, str) else None
        category = row.get("category")
        if source_type == "bounding box" and category == "point": task = "point_box"
        if source_type == "bounding box" and category == "reference": task = "text_box"
        source_id = _source_id(row)
        raw_hash = digest(raw)
        annotation_id = f"bentxt:{digest([source_revision, source_id, raw_hash])}"
        source_ids[source_id].add(raw_hash) if source_id else None
        input_value, output_value = row.get("input"), row.get("output")
        box, point, geometry_frame, geometry_issues = parse_geometry(row)
        ref_match = _REF.search(input_value) if task == "text_box" and isinstance(input_value, str) else None
        metadata_dependency = classify_metadata_dependency(task, input_value if isinstance(input_value, str) else None,
                                                          output_value if isinstance(output_value, str) else None,
                                                          category)
        record = {
            "schema_version": SCHEMA_VERSION, "annotation_id": annotation_id,
            "image_id": image["area_id"] if image else None,
            "match_status": match_status,
            "optical_identity": optical, "sar_identity": sar,
            "split": image["split"] if image else None,
            "task_type": task, "source_task_type": source_type,
            "source_category": category, "source_dataset": "BigEarthNet.txt",
            "source_record_id": source_id, "source_partition": row.get("split"),
            "use_partition": None,
            "question": input_value if task in {"binary_qa", "multiple_choice_qa"} else None,
            "answer": output_value if task in {"binary_qa", "multiple_choice_qa"} else None,
            "options": row.get("options"),
            "caption": output_value if task == "caption" else None,
            "box": box, "point": point, "geometry_frame": geometry_frame,
            "referenced_text": ref_match.group(1) if ref_match else (input_value if task in {"text_box", "point_box"} else None),
            "raw_source": raw, "source_record_sha256": raw_hash,
            "metadata_dependency": metadata_dependency,
            "training_eligibility": training_eligibility_for(metadata_dependency),
            "coordinate_convention_status": COORDINATE_CONVENTION_STATUS,
            "provenance": {"source_revision": source_revision,
                           "matching_method": MATCH_VERSION if image else None,
                           "matching_keys": ["patch_id", "s1_name"] if image else None},
            "validation_status": None, "validation_issues": [],
        }
        issues = _validate_record(record) + geometry_issues
        if not image: issues.append(match_status if optical and sar else "missing_match_keys")
        if row.get("split") not in {"train", "validation", "test", "bench", None}:
            issues.append("unknown_source_partition")
        if image: image_partitions[image["area_id"]].add(row.get("split"))
        if image and row.get("split") in {"train", "validation", "test"} and row["split"] != image["split"]:
            audit["source_partition_disagrees_with_experiment"] += 1
            issues.append("source_partition_conflict")
        if image and task in {"binary_qa", "multiple_choice_qa"} and isinstance(input_value, str) and isinstance(output_value, str):
            qa_splits[(input_value, output_value)].add(image["split"])
            image_questions[(image["area_id"], input_value)] += 1
        record["validation_issues"] = sorted(set(issues))
        record["validation_status"] = "quarantined" if issues else "validated"
        records.append(record)
    records.sort(key=lambda r: (r["annotation_id"], r["source_record_sha256"]))
    id_counts = Counter(r["annotation_id"] for r in records)
    raw_counts = Counter(r["source_record_sha256"] for r in records)
    for record in records:
        image_id = record["image_id"]
        if image_id:
            partitions = image_partitions[image_id]
            if "bench" in partitions:
                record["use_partition"] = "bench_holdout"
                record["validation_issues"].append("benchmark_held_out")
            elif partitions != {record["split"]}:
                record["use_partition"] = "split_conflict_holdout"
                record["validation_issues"].append("image_partition_conflict")
            else:
                record["use_partition"] = record["split"]
        else:
            record["use_partition"] = "unmatched_quarantine"
        if id_counts[record["annotation_id"]] > 1:
            record["validation_issues"].append("duplicate_annotation_id")
        if raw_counts[record["source_record_sha256"]] > 1:
            record["validation_issues"].append("duplicate_source_record")
        if record["source_record_id"] and len(source_ids[record["source_record_id"]]) > 1:
            record["validation_issues"].append("duplicate_source_id_conflict")
        if record["validation_issues"]:
            record["validation_issues"] = sorted(set(record["validation_issues"]))
            record["validation_status"] = "quarantined"
    split_audit = {"experiment_annotation_counts": dict(sorted(Counter(r["split"] for r in records if r["split"]).items())),
                   "matched_image_counts": dict(sorted(Counter({r["image_id"]: r["split"] for r in records if r["image_id"]}.values()).items())),
                   "use_partition_annotation_counts": dict(sorted(Counter(r["use_partition"] for r in records).items())),
                   "use_partition_image_counts": dict(sorted(Counter({r["image_id"]: r["use_partition"] for r in records if r["image_id"]}.values()).items())),
                   "source_partition_disagreements": audit["source_partition_disagrees_with_experiment"],
                   "source_partition_is_not_experiment_split": True}
    match_audit = dict(sorted(Counter(r["match_status"] for r in records).items()))
    leakage = {"duplicate_annotation_ids": sum(v - 1 for v in id_counts.values() if v > 1),
               "duplicate_source_records": sum(v - 1 for v in raw_counts.values() if v > 1),
               "duplicate_source_ids_with_different_content": sum(len(v) > 1 for v in source_ids.values()),
               "identical_qa_across_splits": sum(len(v) > 1 for v in qa_splits.values()),
               "identical_qa_across_splits_is_not_image_leakage": True,
               "duplicate_image_question_combinations": sum(v - 1 for v in image_questions.values() if v > 1),
               "image_level_leakage": 0, "duplicate_images_across_splits": 0,
               "benchmark_held_out_records": sum(r["use_partition"] == "bench_holdout" for r in records),
               "filename_answer_leakage": "unverified; no task-specific filename/answer vocabulary",
               "metadata_answer_leakage": "unverified; source metadata semantics require review",
               "reference_box_reuse": "unverified; no grounding training implemented"}
    leakage["match_categories"] = match_audit
    leakage["test_rows_assigned_to_train"] = sum(r["split"] == "test" and r["use_partition"] == "train" for r in records)
    leakage["metadata_answer_risk_task_counts"] = dict(sorted(Counter(r["source_category"] for r in records if r["task_type"] == "multiple_choice_qa" and r["source_category"] in {"country", "season", "climate zone"}).items()))
    leakage["filename_answer_risk"] = "acquisition dates/tile IDs are encoded in optical identity; task-specific inference unverified"
    return records, split_audit, leakage


def write_artifact(source: Path, manifest: Path, split: Path, output: Path, *,
                   source_revision: str | None = None, code_revision: str | None = None,
                   expected_source_sha256: str | None = None, source_url: str | None = None) -> dict:
    source_sha256 = file_digest(source)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError("Text source checksum mismatch")
    index, image_info = load_images(manifest, split)
    eligible_optical = {key[0] for key in index}
    scan = Counter()
    observed_types, observed_categories, observed_partitions, null_fields = Counter(), Counter(), Counter(), Counter()
    if Path(source).suffix == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(source)
        source_columns = [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow]
        declared_rows = parquet.metadata.num_rows
    else:
        source_columns, declared_rows = None, None

    def selected_rows():
        for row in read_source(source):
            scan["scanned_source_records"] += 1
            observed_types[str(row.get("type"))] += 1
            observed_categories[f"{row.get('type')}::{row.get('category')}"] += 1
            observed_partitions[str(row.get("split"))] += 1
            null_fields.update(k for k, v in row.items() if v is None)
            if any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ("patch_id", "s1_name", "input", "output", "type", "split")):
                scan["malformed_required_fields"] += 1
            if isinstance(row.get("patch_id"), str) and row["patch_id"] in eligible_optical:
                yield row
            else:
                scan["outside_selected_image_inventory"] += 1

    records, split_audit, leakage = ingest(selected_rows(), index, source_revision=source_revision)
    if file_digest(source) != source_sha256:
        raise ValueError("Text source changed during ingestion")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    annotation_bytes = b"".join(canonical_bytes(r) + b"\n" for r in records)
    (output / "annotations.jsonl").write_bytes(annotation_bytes)
    metadata_counts = dict(sorted(Counter(r["metadata_dependency"] for r in records).items()))
    training_counts = dict(sorted(Counter(r["training_eligibility"] for r in records).items()))
    report = {"schema_version": SCHEMA_VERSION, "source_records": len(records),
              "scanned_source_records": scan["scanned_source_records"],
              "outside_selected_image_inventory": scan["outside_selected_image_inventory"],
              "malformed_source_rows_required_fields": scan["malformed_required_fields"],
              "valid_source_rows_required_fields": scan["scanned_source_records"] - scan["malformed_required_fields"],
              "matched_images": len({r["image_id"] for r in records if r["image_id"]}),
              "images_without_records": image_info["image_count"] - len({r["image_id"] for r in records if r["image_id"]}),
              "matched_records": sum(r["image_id"] is not None for r in records),
              "unmatched_records": sum(r["image_id"] is None for r in records),
              "validated_records": sum(r["validation_status"] == "validated" for r in records),
              "quarantined_records": sum(r["validation_status"] == "quarantined" for r in records),
              "task_counts": dict(sorted(Counter(r["task_type"] or "unknown" for r in records).items())),
              "metadata_dependency_counts": metadata_counts,
              "training_eligibility_counts": training_counts,
              "issue_counts": dict(sorted(Counter(i for r in records for i in r["validation_issues"]).items())),
              "geometry_check": "Parsed and validated source unit-square strings; axis origin/raster mapping unknown",
              "parsed_boxes": sum(r["box"] is not None for r in records if r["task_type"] in {"text_box", "point_box"}),
              "parsed_points": sum(r["point"] is not None for r in records if r["task_type"] == "point_box"),
              "coordinate_convention_status": COORDINATE_CONVENTION_STATUS}
    provenance = {"source_dataset": "BigEarthNet.txt", "source_revision": source_revision,
                  "source_url": source_url, "source_file_name": Path(source).name,
                  "source_sha256": source_sha256,
                  "schema_version": SCHEMA_VERSION, "matching_method": MATCH_VERSION,
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "code_revision": code_revision,
                  "matching_implementation_sha256": file_digest(Path(__file__)), **image_info,
                  "annotations_sha256": sha256(annotation_bytes).hexdigest()}
    source_schema = {"observed_from_file": True, "source_format": Path(source).suffix.lstrip("."),
                     "source_revision": source_revision, "source_sha256": source_sha256,
                     "declared_rows": declared_rows, "scanned_rows": scan["scanned_source_records"],
                     "columns": source_columns, "column_count": len(source_columns) if source_columns else None,
                     "type_counts": dict(sorted(observed_types.items())),
                     "type_category_counts": dict(sorted(observed_categories.items())),
                     "source_partition_counts": dict(sorted(observed_partitions.items())),
                     "null_field_counts": dict(sorted(null_fields.items()))}
    geometry = {"schema_version": SCHEMA_VERSION, "source_revision": source_revision,
                "box_format": "[first first, second second] source unit-square numeric string",
                "point_format": "<point>(first, second)</point> source unit-square numeric string",
                "parsed_boxes": report["parsed_boxes"], "parsed_points": report["parsed_points"],
                "repeated_identical_point_tags": sum(len(_POINT.findall(r["raw_source"]["input"])) > 1
                                                     for r in records if r["task_type"] == "point_box"),
                "issue_counts": dict(sorted(Counter(i for r in records for i in r["validation_issues"]
                                                   if "box" in i or "point" in i).items())),
                "coordinate_convention_status": COORDINATE_CONVENTION_STATUS,
                "axis_origin": "unknown", "raster_mapping": "unknown", "sensor_grid": "unknown",
                "inclusive_exclusive_boundary": "unknown", "crs": "not established for box/point strings"}
    identity = {"schema_version": SCHEMA_VERSION, "source_revision": source_revision,
                "source_sha256": source_sha256, "dataset_fingerprint": image_info["dataset_fingerprint"],
                "split_fingerprint": image_info["split_fingerprint"], "code_revision": code_revision,
                "matching_implementation_sha256": provenance["matching_implementation_sha256"]}
    for name, payload in (("manifest.json", {"schema_version": SCHEMA_VERSION,
                                              "annotations_file": "annotations.jsonl",
                                              "annotations_sha256": provenance["annotations_sha256"],
                                              "record_count": len(records)}),
                          ("validation_report.json", report), ("split_audit.json", split_audit),
                          ("leakage_audit.json", leakage), ("source_schema.json", source_schema),
                          ("geometry_audit.json", geometry), ("provenance.json", provenance)):
        (output / name).write_bytes(canonical_bytes({**identity, **payload}) + b"\n")
    return report
