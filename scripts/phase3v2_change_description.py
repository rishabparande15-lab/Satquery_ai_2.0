"""Persist Phase 3V.2 evidence without opening LEVIR-CC TEST material."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "temporal" / "phase3v2_change_description"
CHECKPOINT = ROOT / "checkpoints" / "chg2cap" / "LEVIR_CC_batchsize_32_resnet101.pth"
ARCHIVE = ROOT / "checkpoints" / "chg2cap" / "models_checkpoint.zip"


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    sources = {
        "official_dataset_repository": "https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset",
        "official_specialist_repository": "https://github.com/ShizhenChang/Chg2Cap",
        "official_dataset_facts": {"pair_count_reported": 10077, "caption_count_reported": 50385,
                                   "train_paths": ["images/train/A", "images/train/B"],
                                   "validation_paths": ["images/val/A", "images/val/B"],
                                   "temporal_order": "A=PRE, B=POST"},
        "test_access": 0,
        "dataset_terms": "TERMS_UNCLEAR: the official repository documents download and citation but no explicit image-data license was found in its repository materials; the linked Hugging Face mirror declares Apache-2.0 but is not substituted for the source repository's imagery terms.",
    }
    write("phase3v2_dataset_receipt.json", {"timestamp_utc": timestamp, **sources,
        "development_acquisition": "BLOCKED_MONOLITHIC_ARCHIVE_CONTAINS_TEST",
        "reason": "The official readily accessible archive is monolithic; its published preprocessing reads a global caption JSON and emits a test list. Neither archive nor global captions were opened for development."})
    manifest = {"status": "NOT_CREATED", "reason": "No official split-scoped image/caption manifest was available. Creating identifiers from the monolithic archive would breach TEST_ACCESS=0.", "identities": [], "test_access": 0}
    write("phase3v2_train_manifest.json", {**manifest, "split": "train"})
    write("phase3v2_val_manifest.json", {**manifest, "split": "validation"})
    write("phase3v2_specialist_audit.json", {"specialist": "Chg2Cap", "repository": sources["official_specialist_repository"],
        "architecture": "ResNet-101 dual-image encoder; attentive transformer encoder; transformer caption decoder.",
        "preprocessing": "RGB input, published LEVIR mean/std, 256x256 feature grid expected by the supplied model.",
        "checkpoint_file": "LEVIR_CC_batchsize_32_resnet101.pth", "decoding": "decoder.sample greedy decoding",
        "official_evaluator_metrics": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"],
        "stock_test_command_used": False, "stock_test_command_reason": "It is hard-wired to the TEST split."})
    write("phase3v2_pretrained_receipt.json", {"source": "Author-provided models_checkpoint.zip linked by the official Chg2Cap README",
        "archive_sha256": digest(ARCHIVE), "checkpoint_sha256": digest(CHECKPOINT), "checkpoint_loaded_strictly": CHECKPOINT.is_file(),
        "checkpoint_state_dicts": ["encoder_dict", "encoder_trans_dict", "decoder_dict"], "vocabulary_entries_required": 501,
        "pretrained_inference": "BLOCKED_OFFICIAL_VOCABULARY_NOT_AVAILABLE_WITHOUT_OPENING_MONOLITHIC_CAPTIONS",
        "test_access": 0})
    write("phase3v2_validation_results.json", {"status": "NOT_RUN", "panel_count": 0,
        "reason": "No TRAIN/VALIDATION-only LEVIR image and caption material was safely acquired.", "metrics": None, "test_access": 0})
    write("phase3v2_temporal_sanity.json", {"status": "NOT_RUN", "conditions": ["PRE_POST", "POST_PRE", "PRE_PRE", "POST_POST"],
        "reason": "Sanity controls require a legal validation pair and a functioning official vocabulary; no substitute or synthetic benchmark result was used.", "test_access": 0})
    write("phase3v2_controller_results.json", {"route": "TEMPORAL_CHANGE_DESCRIPTION", "controller_registry": "EXPERIMENTAL", "wrapper": "src/temporal_change.py",
        "separation": "The route is distinct from SINGLE_IMAGE_VQA and OPTICAL_SAR_ANALYSIS.", "status": "CONTRACT_TESTED; LIVE_SPECIALIST_NOT_READY", "test_access": 0})
    write("phase3v2_frontend_verification.json", {"ui_mode": "BI-TEMPORAL", "labels": ["T1 / BEFORE", "T2 / AFTER"],
        "api": "POST /api/v1/temporal", "static_ui_test": "PASS", "real_playwright_success_path": "NOT_RUN",
        "reason": "A real success path requires genuine specialist inference and must not be mocked.", "test_access": 0})
    write("phase3v2_resource_profile.json", {"device": "NVIDIA GeForce RTX 5060 Laptop GPU", "model_load": "PASS", "runtime_per_pair": None,
        "panel_runtime": None, "peak_vram": None, "peak_ram": None, "reason": "No caption generation was run.", "test_access": 0})
    write("phase3v2_summary.json", {"phase_status": "PHASE3V2_DATA_BLOCKED", "route_implementation": "INTEGRATION_PARTIAL_FAIL_CLOSED",
        "dataset": "LEVIR_CC", "specialist": "Chg2Cap", "chronology": "A=PRE, B=POST", "test_access": 0,
        "temporal_model_work_authorized": False, "blockers": ["No official split-scoped TRAIN/VALIDATION acquisition", "No TEST-safe official vocabulary/caption manifest", "Dataset image-data terms remain unclear"],
        "next_phase": "PHASE3V2A_LEVIR_CC_DEVELOPMENT_SPLIT_ADMISSION", "timestamp_utc": timestamp})


if __name__ == "__main__":
    main()
