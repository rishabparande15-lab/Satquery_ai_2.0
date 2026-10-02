"""Write Phase 3V.1B contact readiness and read-only fallback admission.

No network, model, test annotation, or dataset-download operation is performed.
Inputs are the prior TRAIN/VALIDATION-only contact evidence bundle.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "artifacts/final/temporal/phase3v1a_authoritative_resolution"
OUT = ROOT / "artifacts/final/temporal/phase3v1b_fallback_admission"
REVISION = "cc5893123dd32326de38745b65d2ffe45055937b"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((PREVIOUS / "phase3v1a_author_contact_bundle.json").read_text(encoding="utf-8"))
    contacts = {
        "cdvqa": {"official_repository": "https://github.com/YZHJessica/CDVQA", "authors_identified": ["Zhenghang Yuan", "Lichao Mou", "Zhitong Xiong", "Xiao Xiang Zhu"], "public_corresponding_author_email": None, "contact_route": "Official repository issue tracker exists, but issue creation is restricted; no verified public email was found in official project materials."},
        "second": {"official_project_page": "https://captain-whu.github.io/SCD/", "professional_contacts": [{"name": "Kunping Yang", "email": "kunpingyang@whu.edu.cn"}, {"name": "Gui-Song Xia", "email": "guisong.xia@whu.edu.cn"}]},
        "email_sent": False, "test_access_count": 0,
    }
    dump("phase3v1b_author_contacts.json", contacts)
    dump("phase3v1b_author_email_receipt.json", {"subject": "Clarification request: CDVQA–SECOND image mapping and temporal ordering", "ready": True, "sent": False, "recipient_binding": "SECOND professional contacts are verified; no CDVQA corresponding-author email is verified in official sources.", "revision": REVISION, "train_examples": bundle["train_examples"], "validation_examples": bundle["validation_examples"], "test_access_count": 0})
    sources = {
        "cdvqa": {"repository": "https://github.com/YZHJessica/CDVQA", "paper": "https://arxiv.org/abs/2112.06343", "task": "change visual question answering", "source_status": "annotation source admitted; imagery linkage awaiting author confirmation"},
        "qag360k": {"repository": "https://github.com/like413/VisTA", "project_page": "https://like413.github.io/CDQAG/", "paper": "https://arxiv.org/abs/2410.23828", "download": "official Google Drive link from repository README"},
        "cdchat": {"repository": "https://github.com/techmn/cdchat", "paper": "https://arxiv.org/abs/2409.16261", "task": "change description / instruction data"},
        "levircc": {"repository": "https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset", "task": "change captioning and change detection"},
    }
    dump("phase3v1b_dataset_sources.json", sources)
    qag = {
        "dataset": "QAG-360K", "task": {"change_vqa": True, "change_captions": False, "masks": True, "basis": "Official source describes question, textual-answer, and visual-mask triplets."},
        "temporal_order": {"explicit": False, "finding": "Official material establishes paired images from different periods, but this audit did not find a documented pre/post image-path convention."},
        "schema": "Not inspected: no dataset payload was downloaded or opened.", "splits": {"train": "UNVERIFIED", "validation": "UNVERIFIED", "test": "official README says testing set requires contact", "test_isolated_for_development": "UNVERIFIED"},
        "license": {"code": "CC BY-NC 4.0", "dataset_use": "research-only; do not distribute", "underlying_imagery": "mixed sources; per-source terms require separate audit"},
        "official_code": True, "official_loader": True, "official_evaluator": "partial: official benchmark code exists; metric/evaluator protocol was not independently verified in this no-download audit", "acquisition": "UNKNOWN", "admission": "TEMPORAL_DATASET_BLOCKED_ORDER", "not_adopted": True,
    }
    cdchat = {
        "dataset": "CDChat instruction data", "task": {"change_vqa": False, "change_captions": True, "masks": False, "basis": "Official repository calls the task change description and documents text/image-pair instruction data."},
        "temporal_order": {"explicit": False}, "schema": "Not inspected; no payload download.", "splits": {"train": "instruction data listed", "validation": "UNVERIFIED", "test": "evaluation test files listed", "test_isolated_for_development": "UNVERIFIED"},
        "license": {"code": "not stated in official README", "annotations": "UNVERIFIED", "imagery": "depends on SYSU-CD/LEVIR-CD; not separately admitted"},
        "official_code": True, "official_loader": True, "official_evaluator": "YES for change-description evaluation script, not a verified Change-VQA evaluator", "acquisition": "UNKNOWN", "admission": "TEMPORAL_DATASET_NOT_VQA", "not_adopted": True,
    }
    levir = {
        "dataset": "LEVIR-CC", "task": {"change_vqa": False, "change_captions": True, "masks": False, "basis": "Official README describes sentences/captions, not questions with authoritative answers."},
        "temporal_order": {"explicit": True, "rule": "A contains pre-phase images; B contains post-phase images."},
        "schema": {"captions": "LevirCCcaptions.json", "paths": "images/{train,val,test}/{A,B}"}, "splits": {"train": "YES", "validation": "YES", "test": "YES", "test_isolated_for_development": "directory-level split exists, but source terms still need admission"},
        "license": {"code": "UNVERIFIED", "annotations": "UNVERIFIED", "imagery": "UNVERIFIED"},
        "official_code": True, "official_loader": "README documents layout", "official_evaluator": "caption metrics are task-specific; not a Change-VQA evaluator", "acquisition": "download sources documented, but scoped terms were not established", "admission": "TEMPORAL_DATASET_NOT_VQA", "not_adopted": True,
    }
    additional = {"candidates": [{"name": "No additional fully qualifying authoritative Change-VQA dataset was identified in this audit.", "reason": "QAG-360K is the credible QA candidate, but temporal ordering and split-safe acquisition are not yet established."}], "test_access_count": 0}
    dump("phase3v1b_qag360k_audit.json", qag)
    dump("phase3v1b_cdchat_audit.json", cdchat)
    dump("phase3v1b_levircc_audit.json", levir)
    dump("phase3v1b_additional_candidates.json", additional)
    comparison = {
        "columns": ["dataset", "change_vqa", "change_captions", "masks", "explicit_t1_t2", "train", "validation", "test_isolated", "license", "official_loader", "official_evaluator", "acquisition_safe", "status"],
        "rows": [
            ["CDVQA", "YES", "NO", "not in admitted payload", "NO", "YES", "YES", "YES: annotations only", "annotations Apache-2.0; imagery unclear", "NO", "OA/AA defined; runnable evaluator NO", "NO", "CDVQA_AWAITING_AUTHOR_CONFIRMATION"],
            ["QAG-360K", "YES", "NO", "YES", "NO", "UNVERIFIED", "UNVERIFIED", "UNVERIFIED", "research-only / CC BY-NC; mixed imagery terms", "YES", "partial/unverified", "NO", qag["admission"]],
            ["CDChat", "NO", "YES", "NO", "NO", "partial", "UNVERIFIED", "UNVERIFIED", "terms unclear", "YES", "description evaluator", "NO", cdchat["admission"]],
            ["LEVIR-CC", "NO", "YES", "NO", "YES: A=pre, B=post", "YES", "YES", "directory split", "terms unclear", "README layout", "caption-only", "NO", levir["admission"]],
        ], "test_access_count": 0,
    }
    dump("phase3v1b_comparison.json", comparison)
    decision = {"cdvqa": "CDVQA_AWAITING_AUTHOR_CONFIRMATION", "qag360k": qag["admission"], "cdchat": cdchat["admission"], "levircc": levir["admission"], "preferred_temporal_fallback": "NONE", "primary_temporal_dataset": None, "policy_case": "CASE 3: no dataset clears all mandatory gates", "temporal_dataset_work_remains_blocked": True, "test_access_count": 0}
    dump("phase3v1b_admission_decision.json", decision)
    dump("phase3v1b_summary.json", {"status": "PHASE3V1B_AWAITING_AUTHOR_ONLY", "cdvqa": decision["cdvqa"], "preferred_fallback": "NONE", "author_contact_found": True, "email_ready": True, "email_sent": False, "test_access_count": 0, "downloads_performed": False, "training_performed": False, "temporal_model_work_authorized": False, "next_phase": "USER_AUTHORIZED_AUTHOR_OUTREACH_OR_AUTHOR_RESPONSE_REVIEW"})
    print("PHASE3V1B START")
    print("CDVQA STATUS = AWAITING AUTHOR CONFIRMATION")
    print("AUTHOR CONTACT FOUND = YES")
    print("EMAIL READY = YES")
    for name, audit in (("QAG-360K", qag), ("CDChat", cdchat), ("LEVIR-CC", levir)):
        print(f"DATASET = {name}")
        print(f"CHANGE-VQA = {'YES' if audit['task']['change_vqa'] else 'NO'}")
        print(f"T1/T2 EXPLICIT = {'YES' if audit['temporal_order']['explicit'] else 'NO'}")
        print(f"TRAIN/VAL SAFE = NO")
        print(f"LICENSE = {audit['license']}")
        print(f"OFFICIAL EVALUATOR = {audit['official_evaluator']}")
        print(f"ACQUISITION = {audit['acquisition']}")
        print(f"ADMISSION = {audit['admission']}")
    print("CDVQA = CDVQA_AWAITING_AUTHOR_CONFIRMATION")
    print(f"QAG-360K = {qag['admission']}")
    print(f"CDChat = {cdchat['admission']}")
    print(f"LEVIR-CC = {levir['admission']}")
    print("PREFERRED FALLBACK = NONE")
    print("TEST ACCESS = 0")
    print("TEMPORAL MODEL WORK AUTHORIZED = NO")


if __name__ == "__main__":
    main()
