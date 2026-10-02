"""Record Phase 3V.1C contact readiness without sending any message."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/final/temporal/phase3v1c_author_outreach"
EMAIL = ROOT / "docs/SECOND_CDVQA_AUTHOR_EMAIL_FINAL.md"
PACKAGE = ROOT / "docs/SECOND_CDVQA_AUTHOR_QUESTIONS.md"
REVISION = "cc5893123dd32326de38745b65d2ffe45055937b"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = (EMAIL, PACKAGE)
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing existing contact material: {absent}")
    email_text, package_text = (path.read_text(encoding="utf-8") for path in required)
    six_questions = [
        "How does CDVQA `file_name` map to physical SECOND image files?",
        "Which SECOND image/folder is the earlier/pre-change image and which is the later/post-change image?",
        "Are CDVQA TRAIN and VALIDATION identities directly recoverable from SECOND filenames?",
        "What usage/license terms apply to SECOND imagery?",
        "Is there a TRAIN/VALIDATION-specific file list or manifest that avoids reserved CDVQA TEST identities?",
        "Are semantic maps/change masks available for those same identities?",
    ]
    package_ok = all(question in package_text for question in six_questions)
    email_ok = all(question in email_text for question in six_questions)
    if not package_ok or not email_ok:
        raise ValueError("Existing contact materials do not contain the required six questions")
    dump("phase3v1c_contact_sources.json", {
        "second_contacts": [
            {"name": "Kunping Yang", "institution": "State Key Lab of LIESMARS and School of Computer Science, Wuhan University", "email": "kunpingyang@whu.edu.cn", "source": "https://captain-whu.github.io/SCD/SCD_files/Semantic_Change_Detection.pdf"},
            {"name": "Gui-Song Xia", "institution": "School of Artificial Intelligence, Wuhan University", "email": "guisong.xia@whu.edu.cn", "source": "https://jszy.whu.edu.cn/xiaguisong/en/index.htm"},
        ],
        "cdvqa_contact": {"status": "NO_VERIFIED_PUBLIC_CORRESPONDING_AUTHOR_EMAIL", "source_checked": "https://github.com/YZHJessica/CDVQA", "safe_contact_route": "Do not invent a recipient; use an author-confirmed route if one is supplied."},
        "test_access_count": 0,
    })
    dump("phase3v1c_final_email.json", {
        "subject": "Clarification request: CDVQA–SECOND image mapping and temporal ordering",
        "source_file": str(EMAIL.relative_to(ROOT)).replace("\\", "/"), "body_sha256": hashlib.sha256(email_text.encode("utf-8")).hexdigest(),
        "contains_exact_six_questions": email_ok, "contains_train_validation_examples_only": True, "cdvqa_revision": REVISION,
        "send_authorization_present": False, "sent": False, "test_access_count": 0,
    })
    matrix = {
        "physical_path_mapping": {"status": "UNRESOLVED", "evidence": "No primary author source maps CDVQA stems to SECOND paths."},
        "t1_t2_chronology": {"status": "UNRESOLVED", "evidence": "No authoritative before/after semantics established."},
        "imagery_use_terms": {"status": "UNRESOLVED", "evidence": "No explicit SECOND imagery license/use statement found."},
        "safe_train_val_acquisition": {"status": "UNRESOLVED", "evidence": "No source-specific TRAIN/VALIDATION file list or scoped archive index."},
        "semantic_change_maps": {"status": "UNRESOLVED", "evidence": "Dataset-level labels documented, but no per-stem mapping."},
        "test_access_count": 0,
    }
    dump("phase3v1c_resolution_matrix.json", matrix)
    dump("phase3v1c_summary.json", {
        "status": "PHASE3V1C_COMPLETE", "classification": "CDVQA_AWAITING_AUTHOR_CONFIRMATION",
        "email_ready": True, "email_sent": False, "author_response_received": False,
        "temporal_model_work_authorized": False, "test_access_count": 0,
        "next_step": "Obtain explicit user authorization before sending the prepared email; otherwise wait for user-provided author response and intake it as evidence.",
    })
    print("PHASE3V1C START")
    print("CONTACT PACKAGE VERIFIED = YES")
    print("EMAIL READY = YES")
    print("EMAIL SENT = NO (no user authorization)")
    print("AUTHOR RESPONSE RECEIVED = NO")
    print("TEST ACCESS = 0")
    print("CLASSIFICATION = CDVQA_AWAITING_AUTHOR_CONFIRMATION")


if __name__ == "__main__":
    main()
