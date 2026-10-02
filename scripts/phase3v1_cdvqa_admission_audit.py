"""Produce the Phase 3V.1 CDVQA TRAIN/VALIDATION-only admission evidence.

The input allow-list is deliberately fixed.  This script must never discover,
open, parse, or derive a manifest for TEST or TEST2 data.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/final/temporal/phase3v1_cdvqa_admission"
SOURCE = OUT / "_audit_source"
REVISION = "cc5893123dd32326de38745b65d2ffe45055937b"
REPOSITORY = "https://github.com/YZHJessica/CDVQA"
ALLOWED = {
    "train": ("Train_images.json", "Train_questions.json", "Train_answers.json"),
    "val": ("Val_images.json", "Val_questions.json", "Val_answers.json"),
}


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(split: str) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    files = ALLOWED[split]
    paths = [SOURCE / name for name in files]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing allow-listed source files: {absent}")
    images_raw, questions_raw, answers_raw = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    images = images_raw["images"]
    questions = questions_raw["questions"]
    answers = answers_raw["answers"]
    expected_fields = {
        "images": ["active", "file_name", "id", "questions_ids", "res_x", "res_y"],
        "questions": ["active", "answers_ids", "date_added", "id", "img_id", "question", "type"],
        "answers": ["active", "answer", "date_added", "id", "question_id"],
    }
    actual_fields = {
        "images": sorted(images[0]),
        "questions": sorted(questions[0]),
        "answers": sorted(answers[0]),
    }
    if actual_fields != expected_fields:
        raise ValueError(f"Unexpected {split} schema: {actual_fields}")
    return {
        "root_keys": {"images": sorted(images_raw), "questions": sorted(questions_raw), "answers": sorted(answers_raw)},
        "fields": actual_fields,
        "images": images,
        "questions": questions,
        "answers": answers,
    }, {
        name: {str(row["id"]): row for row in rows}
        for name, rows in (("images", images), ("questions", questions), ("answers", answers))
    }


def audit_split(split: str, data: dict[str, object], indexed: dict[str, dict[str, object]]) -> dict[str, object]:
    images = data["images"]
    questions = data["questions"]
    answers = data["answers"]
    image_by_id, question_by_id, answer_by_id = indexed["images"], indexed["questions"], indexed["answers"]
    image_ids = [str(row["id"]) for row in images]
    question_ids = [str(row["id"]) for row in questions]
    answer_ids = [str(row["id"]) for row in answers]
    image_filenames = [str(row["file_name"]) for row in images]
    q_to_image_ok = [str(row["img_id"]) in image_by_id for row in questions]
    q_to_answer_ok = [
        all(str(answer_id) in answer_by_id and str(answer_by_id[str(answer_id)]["question_id"]) == str(row["id"])
            for answer_id in row["answers_ids"])
        for row in questions
    ]
    answer_to_question_ok = [str(row["question_id"]) in question_by_id for row in answers]
    image_to_question_ok = [
        all(str(question_id) in question_by_id and str(question_by_id[str(question_id)]["img_id"]) == str(row["id"])
            for question_id in row["questions_ids"])
        for row in images
    ]
    type_counter = Counter(str(row["type"]) for row in questions)
    answer_counter = Counter(str(row["answer"]) for row in answers)
    family_answers: dict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        for answer_id in question["answers_ids"]:
            family_answers[str(question["type"])][str(answer_by_id[str(answer_id)]["answer"])] += 1
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in questions:
        family = str(row["type"])
        if len(examples[family]) < 2:
            examples[family].append({"question_id": row["id"], "question": row["question"]})
    return {
        "split": split.upper(),
        "schema": {"root_keys": data["root_keys"], "fields": data["fields"]},
        "counts": {
            "image_records": len(images),
            "unique_pair_identifiers_from_file_name": len(set(image_filenames)),
            "questions": len(questions),
            "answers": len(answers),
            "unique_question_ids": len(set(question_ids)),
            "unique_answer_ids": len(set(answer_ids)),
            "duplicate_image_record_ids": len(image_ids) - len(set(image_ids)),
            "duplicate_question_ids": len(question_ids) - len(set(question_ids)),
            "duplicate_answer_ids": len(answer_ids) - len(set(answer_ids)),
        },
        "linkage": {
            "questions_with_valid_img_id": f"{sum(q_to_image_ok)}/{len(questions)}",
            "questions_with_valid_answers_ids": f"{sum(q_to_answer_ok)}/{len(questions)}",
            "answers_with_valid_question_id": f"{sum(answer_to_question_ok)}/{len(answers)}",
            "image_records_with_consistent_questions_ids": f"{sum(image_to_question_ok)}/{len(images)}",
            "annotation_pair_reference": "file_name is one identifier only; no separate T1/T2 path is represented",
        },
        "families": dict(sorted(type_counter.items())),
        "family_answer_distributions": {key: dict(sorted(value.items())) for key, value in sorted(family_answers.items())},
        "answer_distribution": dict(sorted(answer_counter.items())),
        "examples": dict(sorted(examples.items())),
        "pair_identifiers": sorted(set(image_filenames)),
        "question_ids": sorted(set(question_ids), key=int),
    }


def manifest(audit: dict[str, object]) -> list[dict[str, object]]:
    # Deterministic annotation-only manifest: physical image paths intentionally
    # stay null until SECOND provides exact permitted payload linkage.
    data = audit["_data"]
    answer_by_id = audit["_indexed"]["answers"]
    image_by_id = audit["_indexed"]["images"]
    rows = []
    for question in sorted(data["questions"], key=lambda row: row["id"]):
        answer_ids = question["answers_ids"]
        rows.append({
            "record_id": f"CDVQA-{audit['split'].upper()}-Q{question['id']}",
            "pair_id": image_by_id[str(question["img_id"])]["file_name"],
            "image_t1_path": None,
            "image_t2_path": None,
            "question_id": question["id"],
            "question": question["question"],
            "answer": answer_by_id[str(answer_ids[0])]["answer"],
            "answer_id": answer_ids[0],
            "question_family": question["type"],
            "split": audit["split"].upper(),
            "source_dataset": "CDVQA",
            "underlying_dataset": "SECOND",
            "source_revision": REVISION,
            "temporal_order": "TEMPORAL_ORDER_UNRESOLVED",
            "provenance": REPOSITORY,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    audited = {}
    file_receipts = []
    for split, files in ALLOWED.items():
        data, indexed = load_split(split)
        loaded[split] = (data, indexed)
        audit = audit_split(split, data, indexed)
        audit["_data"], audit["_indexed"] = data, indexed
        audited[split] = audit
        for name in files:
            path = SOURCE / name
            file_receipts.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})

    train, val = audited["train"], audited["val"]
    train_pairs, val_pairs = set(train["pair_identifiers"]), set(val["pair_identifiers"])
    train_questions, val_questions = set(train["question_ids"]), set(val["question_ids"])
    all_answers = set(train["answer_distribution"]) | set(val["answer_distribution"])
    unseen_val_answers = sorted(set(val["answer_distribution"]) - set(train["answer_distribution"]))
    files = {entry["name"]: entry for entry in file_receipts}

    dump("phase3v1_source_receipt.json", {
        "official_repository": REPOSITORY,
        "revision": REVISION,
        "license": "Apache-2.0 (repository-listed)",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_files_only": sorted(files),
        "files": files,
        "test_payload_access_count": 0,
        "test_or_test2_files_read": [],
    })
    for split, name in (("train", "phase3v1_train_schema.json"), ("val", "phase3v1_val_schema.json")):
        dump(name, {"split": split.upper(), "root_keys": audited[split]["schema"]["root_keys"], "fields": audited[split]["schema"]["fields"]})
    dump("phase3v1_counts.json", {
        "train": train["counts"], "validation": val["counts"],
        "paper_reported": {"train_pairs": 1600, "train_qa": 65967, "validation_pairs": 400, "validation_qa": 16441},
        "comparison": {
            "train_pair_count_matches": train["counts"]["unique_pair_identifiers_from_file_name"] == 1600,
            "train_qa_count_matches": train["counts"]["questions"] == 65967 and train["counts"]["answers"] == 65967,
            "validation_pair_count_matches": val["counts"]["unique_pair_identifiers_from_file_name"] == 400,
            "validation_qa_count_matches": val["counts"]["questions"] == 16441 and val["counts"]["answers"] == 16441,
        },
    })
    dump("phase3v1_question_taxonomy.json", {
        "train": {"counts": train["families"], "answer_distributions": train["family_answer_distributions"], "examples": train["examples"]},
        "validation": {"counts": val["families"], "answer_distributions": val["family_answer_distributions"], "examples": val["examples"]},
        "paper_family_mapping": {
            "change_or_not": "CHANGE_OR_NOT", "change_to_what": "CHANGE_TO_WHAT",
            "increase_or_not": "INCREASE_OR_NOT", "decrease_or_not": "DECREASE_OR_NOT",
            "change_ratio": "CHANGE_RATIO", "change_ratio_types": "CLASS_CHANGE_RATIO",
            "smallest_change": "SMALLEST_CHANGE", "largest_change": "LARGEST_CHANGE",
        },
    })
    dump("phase3v1_answer_space.json", {
        "train_vocabulary": sorted(train["answer_distribution"]), "validation_vocabulary": sorted(val["answer_distribution"]),
        "combined_class_count": len(all_answers), "validation_answers_unseen_in_train": unseen_val_answers,
        "train_distribution": train["answer_distribution"], "validation_distribution": val["answer_distribution"],
    })
    dump("phase3v1_temporal_order_audit.json", {
        "status": "TEMPORAL_ORDER_UNRESOLVED", "finding": "The allowed JSON schema has one file_name per image record and no T1/T2, before/after, A/B, timestamp, or temporal-order field.",
        "safe_internal_value": None, "modeling_authorized": False,
    })
    dump("phase3v1_second_provenance.json", {
        "source": "https://captain-whu.github.io/SCD/", "source_kind": "author-maintained SECOND project page",
        "payload_location": "official project page links to Google Drive", "payload_acquired": False,
        "reason_not_acquired": "The official source exposes no TRAIN/VALIDATION-only CDVQA mapping or license/use terms sufficient to exclude CDVQA TEST identities before download.",
        "documented_properties": {"image_size": "512x512", "classes": ["non-vegetated ground surface", "tree", "low vegetation", "water", "buildings", "playgrounds"], "pixel_level_annotations": True},
        "license_or_use_terms": "not stated on the official project page; contact-based access is listed", "test_payload_access_count": 0,
    })
    dump("phase3v1_image_linkage.json", {
        "annotation_linkage": {"train_questions_to_image_records": train["linkage"]["questions_with_valid_img_id"], "validation_questions_to_image_records": val["linkage"]["questions_with_valid_img_id"]},
        "physical_payload_linkage": {"train_pairs_linked": f"0/{len(train_pairs)}", "validation_pairs_linked": f"0/{len(val_pairs)}", "missing_pairs": len(train_pairs) + len(val_pairs), "ambiguous_pairs": len(train_pairs) + len(val_pairs)},
        "reason": "CDVQA metadata stores a single file_name per image record, but no admitted SECOND payload or exact T1/T2 filename mapping is available.",
        "semantic_change_evidence": "SPATIAL_CHANGE_EVIDENCE_NOT_AVAILABLE_IN_ADMITTED_PAYLOAD",
    })
    dump("phase3v1_split_audit.json", {
        "train_pair_ids_intersect_validation_pair_ids": sorted(train_pairs & val_pairs),
        "train_pair_overlap_count": len(train_pairs & val_pairs),
        "train_image_files_intersect_validation_image_files": sorted(train_pairs & val_pairs),
        "train_question_id_numeric_overlap_count": len(train_questions & val_questions),
        "question_id_interpretation": "IDs restart per split; numeric overlap is expected and not evidence of paired-image leakage.",
        "physical_file_overlap_verifiable": False, "test_payload_access_count": 0,
    })
    dump("phase3v1_metric_audit.json", {
        "task": "closed answer classification", "official_reporting": ["Overall Accuracy (OA)", "Average Accuracy (AA)"],
        "definition": {"OA": "correct answer predictions divided by all evaluated question-answer records", "AA": "arithmetic mean of the eight per-question-family accuracies"},
        "evaluator_release": "No runnable official evaluator was found in the official annotation repository audit.",
    })
    dump("phase3v1_baseline_audit.json", {
        "paper_architecture": ["multi-temporal feature encoding", "multi-temporal fusion", "multimodal fusion with question", "answer prediction"],
        "change_enhancing_module": "described in the paper as an addition to multi-temporal feature encoding", "runnable_baseline": "OFFICIAL_RUNNABLE_BASELINE_NOT_RELEASED_HERE", "checkpoint": None,
    })
    dump("phase3v1_train_manifest.json", {"kind": "annotation-only deterministic manifest", "records": manifest(train), "physical_payload_admitted": False})
    dump("phase3v1_val_manifest.json", {"kind": "annotation-only deterministic manifest", "records": manifest(val), "physical_payload_admitted": False})
    classification = "CDVQA_ANNOTATIONS_ADMITTED_IMAGES_PENDING"
    dump("phase3v1_summary.json", {
        "status": "PHASE3V1_COMPLETE", "dataset_classification": classification, "temporal_model_work_authorized": False,
        "blocking_conditions": ["TEMPORAL_ORDER_UNRESOLVED", "physical TRAIN/VALIDATION imagery unavailable and unlinked", "SECOND license/use terms not established for scoped acquisition"],
        "test_payload_access_count": 0, "training_performed": False,
        "next_step": "Obtain authoritative SECOND TRAIN/VALIDATION-only T1/T2 linkage, temporal-order documentation, and scoped use authorization before downloading any imagery.",
    })
    print("PHASE3V1 START")
    print(f"OFFICIAL CDVQA COMMIT = {REVISION}")
    print("LICENSE = Apache-2.0 (repository-listed)")
    print("TRAIN FILES VERIFIED = 3/3")
    print("VAL FILES VERIFIED = 3/3")
    print("TEST PAYLOAD ACCESS = 0")
    print(f"TRAIN IMAGE PAIRS = {len(train_pairs)}")
    print(f"TRAIN QUESTIONS = {train['counts']['questions']}")
    print(f"TRAIN ANSWERS = {train['counts']['answers']}")
    print(f"VAL IMAGE PAIRS = {len(val_pairs)}")
    print(f"VAL QUESTIONS = {val['counts']['questions']}")
    print(f"VAL ANSWERS = {val['counts']['answers']}")
    print(f"QUESTION FAMILIES = {', '.join(sorted(train['families']))}")
    print(f"ANSWER CLASSES = {len(all_answers)}")
    print("TEMPORAL ORDER = TEMPORAL_ORDER_UNRESOLVED")
    print("SECOND IMAGE SOURCE = author-maintained SECOND project page -> Google Drive")
    print(f"TRAIN IMAGE LINKAGE = 0/{len(train_pairs)} physical; {train['linkage']['questions_with_valid_img_id']} annotation")
    print(f"VAL IMAGE LINKAGE = 0/{len(val_pairs)} physical; {val['linkage']['questions_with_valid_img_id']} annotation")
    print(f"TRAIN-VAL OVERLAP = {len(train_pairs & val_pairs)} pair IDs")
    print("OFFICIAL METRIC = OA and AA")
    print(f"DATASET CLASSIFICATION = {classification}")


if __name__ == "__main__":
    main()
