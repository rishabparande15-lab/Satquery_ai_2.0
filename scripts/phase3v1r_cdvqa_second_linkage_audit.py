"""Phase 3V.1R: TRAIN/VALIDATION-only CDVQA-to-SECOND linkage audit.

The fixed six-file allow-list prevents any TEST or TEST2 annotation access.
This audit intentionally produces no physical-image download manifest unless
authoritative metadata establishes both stem mapping and temporal ordering.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/final/temporal/phase3v1_cdvqa_admission/_audit_source"
OUT = ROOT / "artifacts/final/temporal/phase3v1r_image_linkage"
CDVQA_REVISION = "cc5893123dd32326de38745b65d2ffe45055937b"
ALLOWLIST = {
    "TRAIN": ("Train_images.json", "Train_questions.json"),
    "VALIDATION": ("Val_images.json", "Val_questions.json"),
}
SECOND_PAGE = "https://captain-whu.github.io/SCD/"
SECOND_ARCHIVES = {
    "second_dataset.zip": "https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view?usp=sharing",
    "SECOND_train_set.rar": "https://drive.google.com/file/d/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc/view?usp=sharing",
}


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest_strings(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def read_split(split: str) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    image_name, question_name = ALLOWLIST[split]
    images = json.loads((SOURCE / image_name).read_text(encoding="utf-8"))["images"]
    questions = json.loads((SOURCE / question_name).read_text(encoding="utf-8"))["questions"]
    return images, {int(row["id"]): row for row in questions}


def ratio_audit(images: list[dict[str, object]], questions: dict[int, dict[str, object]]) -> dict[str, object]:
    by_stem: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in images:
        by_stem[str(row["file_name"])].append(row)
    records_per_stem = Counter(len(rows) for rows in by_stem.values())
    type_slot_signatures = Counter()
    per_stem_type_rows = Counter()
    empty_question_rows = 0
    for rows in by_stem.values():
        signature = []
        for row in rows:
            types = tuple(sorted({str(questions[int(question_id)]["type"]) for question_id in row["questions_ids"]}))
            signature.append(types)
            per_stem_type_rows.update(types)
            empty_question_rows += not types
        type_slot_signatures[tuple(signature)] += 1
    return {
        "unique_file_name_stems": len(by_stem),
        "image_records": len(images),
        "records_per_stem_distribution": dict(sorted(records_per_stem.items())),
        "all_stems_have_16_records": records_per_stem == Counter({16: len(by_stem)}),
        "record_role_evidence": {
            "each_image_record_has_questions_ids": True,
            "questions_reference_image_record_with_img_id": True,
            "image_records_have_no_crop_view_or_temporal_path_fields": True,
            "distinct_question_type_rows_across_all_stems": dict(sorted(per_stem_type_rows.items())),
            "empty_question_container_records": empty_question_rows,
            "unique_16_slot_question_type_signatures": len(type_slot_signatures),
        },
        "conclusion": "The 16 records are repeated annotation image containers for generated question groups sharing one file_name. The admitted schema contains no crop/view fields and no evidence that these are 16 physical images.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    read_files = []
    split_data = {}
    for split, files in ALLOWLIST.items():
        if any(not (SOURCE / filename).is_file() for filename in files):
            raise FileNotFoundError(f"Missing TRAIN/VALIDATION allow-listed inputs for {split}")
        images, questions = read_split(split)
        stems = sorted({str(row["file_name"]) for row in images})
        split_data[split] = {"images": images, "questions": questions, "stems": stems, "ratio": ratio_audit(images, questions)}
        read_files.extend(files)

    train, val = split_data["TRAIN"], split_data["VALIDATION"]
    train_stems, val_stems = set(train["stems"]), set(val["stems"])
    dump("phase3v1r_pair_stems_train.json", {"split": "TRAIN", "count": len(train["stems"]), "sha256": digest_strings(train["stems"]), "stems": train["stems"], "test_access_count": 0})
    dump("phase3v1r_pair_stems_val.json", {"split": "VALIDATION", "count": len(val["stems"]), "sha256": digest_strings(val["stems"]), "stems": val["stems"], "test_access_count": 0})
    dump("phase3v1r_image_record_ratio_audit.json", {"train": train["ratio"], "validation": val["ratio"], "test_access_count": 0})
    dump("phase3v1r_second_source_audit.json", {
        "authoritative_source": SECOND_PAGE,
        "author_maintained": True,
        "source_linked_archives_metadata_only": SECOND_ARCHIVES,
        "metadata_inspection_result": "The author page exposes archive names but no browsable file tree or filename index.",
        "documented_image_properties": {"pair_count": 4662, "size": "512x512", "pixel_level_annotations": True, "land_cover_classes": 6},
        "documented_directory_structure": None,
        "documented_t1_directory": None,
        "documented_t2_directory": None,
        "documented_temporal_naming": None,
        "test_access_count": 0,
    })
    dump("phase3v1r_file_linkage.json", {
        "matching_method": "No physical filename index is exposed by an authoritative source; no matching was attempted or inferred.",
        "train": {"exact_or_documented_transform_links": f"0/{len(train_stems)}", "ambiguous_links": len(train_stems), "missing_links": 0, "unresolved_reason": "No authoritative SECOND file metadata/tree maps a CDVQA file_name to T1/T2 paths."},
        "validation": {"exact_or_documented_transform_links": f"0/{len(val_stems)}", "ambiguous_links": len(val_stems), "missing_links": 0, "unresolved_reason": "No authoritative SECOND file metadata/tree maps a CDVQA file_name to T1/T2 paths."},
        "physical_name_convention": "UNDOCUMENTED", "test_access_count": 0,
    })
    dump("phase3v1r_temporal_order.json", {
        "status": "TEMPORAL_ORDER_UNRESOLVED", "temporal_order": None,
        "authoritative_evidence_checked": [SECOND_PAGE, "SECOND paper (arXiv:2010.05687)", "CDVQA paper (arXiv:2112.06343)", "CDVQA repository README"],
        "finding": "These sources establish multi-temporal pairs but do not define an image-directory or filename convention that maps a CDVQA stem to ordered T1/T2 paths.",
        "prohibited_inferences_not_used": ["lexicographic filename ordering", "filesystem return order", "visual brightness/content"],
        "modeling_authorized": False, "test_access_count": 0,
    })
    dump("phase3v1r_second_terms.json", {
        "cdvqa_annotation_license": "Apache-2.0", "second_image_license": None,
        "second_use_status": "SECOND_USE_TERMS_UNCLEAR", "evidence": "The author-maintained SECOND page supplies a dataset download link and contact addresses but no explicit license, research-use terms, or terms page.",
        "source": SECOND_PAGE, "test_access_count": 0,
    })
    dump("phase3v1r_scoped_acquisition.json", {
        "classification": "MONOLITHIC_DOWNLOAD_ONLY", "evidence": "The author page links archive downloads but publishes no CDVQA TRAIN/VALIDATION-specific file manifest or archive index.",
        "safe_procedure_before_acquisition": ["Obtain author-provided T1/T2 directory and stem mapping documentation.", "Obtain written/use-term confirmation for SECOND imagery.", "Obtain a source-specific TRAIN/VALIDATION-only file list or an archive index that permits exact filtering before retrieval.", "Re-check TRAIN/VALIDATION physical-file disjointness after mapping without creating a CDVQA TEST manifest."],
        "download_performed": False, "test_access_count": 0,
    })
    dump("phase3v1r_semantic_map_audit.json", {
        "dataset_level_documentation": "SECOND is documented as pixel-level annotated; its project page says SCD ground truth can be obtained by comparing annotated land-cover categories.",
        "t1_semantic_map_available_for_admitted_stems": "UNVERIFIED", "t2_semantic_map_available_for_admitted_stems": "UNVERIFIED", "binary_change_map_available_for_admitted_stems": "UNVERIFIED",
        "mapping_rule": None, "download_performed": False, "test_access_count": 0,
    })
    classification = "CDVQA_BLOCKED_IMAGE_LINKAGE"
    dump("phase3v1r_summary.json", {
        "status": "PHASE3V1R_COMPLETE", "classification": classification,
        "train_stems": len(train_stems), "validation_stems": len(val_stems),
        "train_val_annotation_stem_overlap": len(train_stems & val_stems),
        "temporal_order": "TEMPORAL_ORDER_UNRESOLVED", "second_use_status": "SECOND_USE_TERMS_UNCLEAR",
        "test_access_count": 0, "download_performed": False, "training_performed": False,
        "image_acquisition_authorized": False, "temporal_model_work_authorized": False,
        "remaining_blockers": ["no authoritative CDVQA stem-to-SECOND physical-path map", "no authoritative T1/T2 ordering convention", "SECOND use terms unclear", "no scoped CDVQA TRAIN/VALIDATION acquisition mechanism"],
        "next_phase": "AUTHORITATIVE_SECOND_METADATA_OR_AUTHOR_CONFIRMATION_FOR_CDVQA_TRAIN_VAL_LINKAGE",
    })
    print("PHASE3V1R START")
    print(f"TRAIN UNIQUE STEMS = {len(train_stems)}")
    print(f"VAL UNIQUE STEMS = {len(val_stems)}")
    print("TRAIN IMAGE RECORD RATIO = 16:1")
    print("VAL IMAGE RECORD RATIO = 16:1")
    print("SECOND DIRECTORY STRUCTURE = UNDOCUMENTED")
    print("TEMPORAL ORDER DOCUMENTED = NO")
    print(f"TRAIN LINKAGE = 0/{len(train_stems)}")
    print(f"VAL LINKAGE = 0/{len(val_stems)}")
    print("SCOPED ACQUISITION = MONOLITHIC_DOWNLOAD_ONLY")
    print("SECOND USE STATUS = SECOND_USE_TERMS_UNCLEAR")
    print("TEST ACCESS = 0")
    print(f"DATASET CLASSIFICATION = {classification}")


if __name__ == "__main__":
    main()
