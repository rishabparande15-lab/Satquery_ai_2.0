"""Write the Phase 3V.1A author-controlled source-resolution record.

The script reads only prior TRAIN/VALIDATION stem artifacts and contains no
network or dataset-download operation.  TEST/TEST2 content is never loaded.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "artifacts/final/temporal/phase3v1r_image_linkage"
OUT = ROOT / "artifacts/final/temporal/phase3v1a_authoritative_resolution"
REVISION = "cc5893123dd32326de38745b65d2ffe45055937b"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = json.loads((PREVIOUS / "phase3v1r_pair_stems_train.json").read_text(encoding="utf-8"))
    val = json.loads((PREVIOUS / "phase3v1r_pair_stems_val.json").read_text(encoding="utf-8"))
    bundle = {
        "cdvqa_revision": REVISION,
        "train_examples": train["stems"][:3],
        "validation_examples": val["stems"][:3],
        "schema_summary": {
            "image_record": ["id", "questions_ids", "file_name"],
            "question_record": ["id", "img_id", "type", "question", "answers_ids"],
            "answer_record": ["id", "question_id", "answer"],
            "verified_annotation_relationship": "question.img_id -> image.id -> image.file_name",
        },
        "unresolved": ["physical SECOND T1/T2 path mapping", "chronological meaning of pair members", "SECOND usage terms", "TRAIN/VALIDATION-only acquisition procedure"],
        "test_access_count": 0,
    }
    dump("phase3v1a_source_search.json", {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sources": [
            {"source": "https://github.com/YZHJessica/CDVQA", "finding": "Official repository contains LICENSE, README, and annotation JSON files; no loader, path-construction code, dataset documentation, or image archive metadata is published."},
            {"source": "https://captain-whu.github.io/SCD/", "finding": "Author-maintained SECOND page documents multi-temporal 512x512 pixel-level data and links archives, but not directory layout, stem mapping, chronology, or terms."},
            {"source": "https://arxiv.org/abs/2112.06343", "finding": "CDVQA paper establishes a pair-based task but does not publish a file-path or T1/T2 naming convention."},
            {"source": "https://arxiv.org/abs/2010.05687", "finding": "SECOND paper establishes multi-temporal semantic change data but does not establish CDVQA stem-to-path linkage in accessible documentation."},
        ],
        "supporting_official_benchmark_source": {"source": "https://github.com/like413/VisTA/blob/main/VisTA/dataset/DataNew.py", "finding": "Lines 568-578 construct im1/<file_name> and im2/<file_name>. This is supporting convention evidence only: it does not identify im1/im2 chronology and is not a primary CDVQA/SECOND release."},
        "test_access_count": 0,
    })
    dump("phase3v1a_loader_code_audit.json", {
        "cdvqa_primary_loader": "NOT_FOUND", "second_primary_loader": "NOT_FOUND", "cdvqa_repository_basis": "The official root file listing contains annotation files, LICENSE, and README only.",
        "second_page_basis": "The author page marks ASN code as coming soon and exposes no dataset loader.",
        "supporting_loader_rule": {"repository": "like413/VisTA (official VisTA benchmark implementation)", "file": "VisTA/dataset/DataNew.py", "lines": "568-578", "rule": "question.img_id is used as file_name; files are read from img_path + 'im1/' + file_name and img_path + 'im2/' + file_name.", "chronology_defined": False, "sufficient_for_cdvqa_admission": False},
        "test_access_count": 0,
    })
    dump("phase3v1a_issue_discussion_audit.json", {
        "source": "https://github.com/YZHJessica/CDVQA/issues", "issues_checked": [
            {"number": 1, "topic": "general project information", "maintainer_response": False, "linkage_or_order_evidence": False},
            {"number": 2, "topic": "date_added and resolution metadata", "maintainer_response": False, "linkage_or_order_evidence": False},
        ], "direct_author_confirmation_found": False, "test_access_count": 0,
    })
    dump("phase3v1a_archive_metadata.json", {
        "author_source": "https://captain-whu.github.io/SCD/", "published_archive_metadata": [
            {"name": "second_dataset.zip", "provider": "Google Drive", "content_inspected": False},
            {"name": "SECOND_train_set.rar", "provider": "Google Drive", "content_inspected": False},
        ], "authoritative_folder_listing_available": False, "t1_t2_directory_documented": False,
        "semantic_directory_documented": False, "reason": "The provider pages expose archive names but no archive manifests or browsable file listings.", "test_access_count": 0,
    })
    dump("phase3v1a_linkage_attempt.json", {
        "primary_authoritative_rule": None, "train_exact_links": "0/1600", "validation_exact_links": "0/400",
        "classification": "AMBIGUOUS", "supporting_unconfirmed_pattern": "im1/<file_name> plus im2/<file_name> in official VisTA loader code", "fuzzy_matching_used": False,
        "conclusion": "No authoritative CDVQA/SECOND source was found that deterministically maps the admitted stems to physical files.", "test_access_count": 0,
    })
    dump("phase3v1a_temporal_order_evidence.json", {
        "status": "TEMPORAL_ORDER_UNRESOLVED", "acceptable_primary_evidence_found": False,
        "checked": ["SECOND project page", "SECOND paper", "CDVQA repository", "CDVQA paper", "CDVQA issues"],
        "supporting_but_insufficient": "VisTA names pair inputs im1 and im2 but does not define before/after or T1/T2 semantics.",
        "inferences_not_used": ["directory lexical order", "filename order", "visual appearance", "filesystem order"], "test_access_count": 0,
    })
    dump("phase3v1a_second_terms.json", {
        "status": "SECOND_USE_TERMS_UNCLEAR", "license": None, "research_use_statement": None, "redistribution_rule": None,
        "evidence": "No explicit SECOND imagery license or use-terms page was found in the author-maintained page, papers, archive metadata, CDVQA repository, or official issue history.",
        "test_access_count": 0,
    })
    dump("phase3v1a_author_contact_bundle.json", bundle)
    dump("phase3v1a_alternative_dataset_audit.json", {
        "trigger": "Primary CDVQA/SECOND linkage remains unresolved.",
        "candidates": [
            {"name": "QAG-360K", "official_source": "https://github.com/like413/VisTA", "paired_temporal_imagery": "YES", "qa_supervision": "YES: question, textual answer, and mask triplets", "temporal_order": "PAIR INPUTS DOCUMENTED, BUT THIS AUDIT DID NOT ESTABLISH A PRE/POST DIRECTORY semantic", "train_val_split": "UNVERIFIED_IN_THIS_READ_ONLY_AUDIT", "use_terms": "CC BY-NC 4.0; research-only; no redistribution", "status": "POTENTIAL_FUTURE_ADMISSION_AUDIT_ONLY; NOT_ADOPTED"},
            {"name": "LEVIR-CC", "official_source": "https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset", "paired_temporal_imagery": "YES; A=pre-phase and B=post-phase", "qa_supervision": "NO: change captions, not question-answer supervision", "train_val_split": "YES", "status": "REJECTED_FOR_THIS_FALLBACK_REQUIREMENT"},
            {"name": "CDChat instruction data", "official_source": "https://github.com/techmn/cdchat", "paired_temporal_imagery": "YES", "qa_supervision": "instruction/change-description data, not independently verified as a split-safe Change-VQA benchmark", "temporal_order": "UNVERIFIED", "use_terms": "UNVERIFIED", "status": "NOT_ADMITTED"},
        ], "adopted_dataset": None, "test_access_count": 0,
    })
    classification = "CDVQA_AWAITING_AUTHOR_CONFIRMATION"
    dump("phase3v1a_summary.json", {
        "status": "PHASE3V1A_COMPLETE", "classification": classification,
        "train_linkage": "0/1600", "validation_linkage": "0/400", "temporal_order": "TEMPORAL_ORDER_UNRESOLVED", "second_terms": "SECOND_USE_TERMS_UNCLEAR",
        "safe_train_val_acquisition": False, "author_clarification_required": True, "test_access_count": 0,
        "temporal_model_work_authorized": False, "training_performed": False, "downloads_performed": False,
        "next_step": "Send the prepared author-contact questions to the CDVQA/SECOND maintainers and wait for a source-backed mapping, chronological ordering, usage terms, and a scope-safe TRAIN/VALIDATION file list.",
    })
    print("PHASE3V1A START")
    print("PRIMARY LOADER/PATH RULE = NOT FOUND")
    print("TRAIN LINKAGE = 0/1600")
    print("VAL LINKAGE = 0/400")
    print("TEMPORAL ORDER = TEMPORAL_ORDER_UNRESOLVED")
    print("SECOND TERMS = SECOND_USE_TERMS_UNCLEAR")
    print("SAFE TRAIN/VAL ACQUISITION = NO")
    print("AUTHOR CLARIFICATION REQUIRED = YES")
    print("TEST ACCESS = 0")
    print(f"CLASSIFICATION = {classification}")


if __name__ == "__main__":
    main()
