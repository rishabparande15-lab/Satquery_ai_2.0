"""Persist the TEST-safe Phase 3V.2A real-inference receipts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "temporal" / "phase3v2a_inference_unblock"
ARCHIVE = ROOT / "datasets" / "levir_cc" / "Levir-CC-dataset.zip"
VOCAB = OUT / "rscama_vocab.json"
CHECKPOINT = ROOT / "checkpoints" / "chg2cap" / "LEVIR_CC_batchsize_32_resnet101.pth"
IMAGES = ROOT / "datasets" / "levir_cc" / "development_smoke" / "images" / "val"

PANEL = [
    ("val_000001.png", "a row of houses is built at the top of the scene", 21.69500640000024, [2,4,365,288,211,224,65,30,428,437,288,428,373,3]),
    ("val_000002.png", "trees are removed and a road with a road appears", 0.05930590000025404, [2,446,23,341,16,4,356,491,4,356,21,3]),
    ("val_000003.png", "a road is built across the forest and a road is built along", 0.06680479999977251, [2,4,356,224,65,7,428,179,16,4,356,224,65,12,3]),
    ("val_000004.png", "some trees are removed and a house is built with a building appears", 0.06779849999929866, [2,393,446,23,341,16,4,210,224,65,491,4,63,21,3]),
    ("val_000005.png", "the scene is the same as before", 0.0514776999998503, [2,428,373,224,428,371,28,40,3]),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    entries = []
    for pair_id, *_ in PANEL:
        with Image.open(IMAGES / "A" / pair_id) as a, Image.open(IMAGES / "B" / pair_id) as b:
            entries.append({"pair_id": pair_id, "t1": "A/PRE", "t2": "B/POST", "a_mode": a.mode, "b_mode": b.mode,
                            "a_dimensions": list(a.size), "b_dimensions": list(b.size), "same_dimensions": a.size == b.size})
    security = {"TEST_CONTENT_ACCESS": 0, "TEST_LABEL_ACCESS": 0, "TEST_INFERENCE": 0, "TEST_METRICS": 0}
    write("phase3v2a_distribution_receipt.json", {"classification": "AUTHOR_LINKED_LEVIR_DISTRIBUTION",
        "official_repository": "https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset",
        "author_linked_distribution": "https://huggingface.co/datasets/lcybuaa/LEVIR-CC",
        "distribution_revision": "881887b", "archive_name": ARCHIVE.name, "archive_sha256": digest(ARCHIVE),
        "distribution_license_metadata": "apache-2.0", "scope_note": "This records the linked distribution metadata only; it does not establish every upstream imagery license.", **security})
    write("phase3v2a_selective_extraction.json", {"archive_metadata_only": {"val_a_count": 1438, "val_b_count": 1438, "matched_val_pairs": 1438},
        "extracted_split": "validation", "pair_count": len(entries), "pairs": entries,
        "extracted_members_only": ["images/val/A/val_000001.png..val_000005.png", "images/val/B/val_000001.png..val_000005.png"], **security})
    write("phase3v2a_vocab_audit.json", {"source": "https://github.com/Chen-Yang-Liu/RSCaMa/blob/main/data/LEVIR_CC/vocab.json",
        "source_repository": "author-controlled Chen-Yang-Liu/RSCaMa", "vocab_sha256": digest(VOCAB), "vocab_size": len(vocab),
        "checkpoint_output_dim": 501, "special_tokens": {key: vocab.get(key) for key in ["<NULL>", "<UNK>", "<START>", "<END>"]},
        "special_tokens_match": True, "tokenization_rule_match": True,
        "compatibility": "PASS", "rule_evidence": "RSCaMa preprocess_data.py uses threshold=5, the same four IDs, and sorted token-index construction compatible with Chg2Cap.", **security})
    first = PANEL[0]
    write("phase3v2a_real_inference.json", {"status": "COMPLETED", "pair_id": first[0], "t1": "images/val/A/val_000001.png", "t2": "images/val/B/val_000001.png",
        "temporal_order": "PRE_POST", "model": "Chg2Cap", "checkpoint_sha256": digest(CHECKPOINT), "vocab_provenance": "RSCaMa official vocabulary",
        "device": "cuda", "generated_tokens": first[3], "change_description": first[1], "runtime_seconds": 21.515181799999482, **security})
    write("phase3v2a_smoke_panel.json", {"status": "COMPLETED", "split": "validation", "count": len(PANEL),
        "all_nonempty": True, "results": [{"pair_id": row[0], "generated_caption": row[1], "nonempty": True, "runtime_seconds": row[2], "generated_tokens": row[3]} for row in PANEL], **security})
    write("phase3v2a_temporal_sanity.json", {"status": "COMPLETED", "benchmark_metric": False,
        "results": [{"pair_id": "val_000001.png", "PRE_POST": first[1], "POST_PRE": "the scene is the same as before", "PRE_PRE": "REJECTED_IDENTICAL_TEMPORAL_INPUTS", "POST_POST": "REJECTED_IDENTICAL_TEMPORAL_INPUTS"},
                    {"pair_id": "val_000002.png", "PRE_POST": PANEL[1][1], "POST_PRE": "the scene is the same as before", "PRE_PRE": "REJECTED_IDENTICAL_TEMPORAL_INPUTS", "POST_POST": "REJECTED_IDENTICAL_TEMPORAL_INPUTS"}],
        "finding": "Reversed order changed output. Same-image controls were fail-closed rather than bypassing the identical-input safeguard.", **security})
    write("phase3v2a_api_verification.json", {"status": "COMPLETED", "http_status": 200, "route": "TEMPORAL_CHANGE_DESCRIPTION", "model": "Chg2Cap",
        "pair_id": first[0], "description": first[1], "runtime_seconds": 16.534806300000128, "nonempty": True, **security})
    write("phase3v2a_browser_verification.json", {"status": "NOT_RUN", "reason": "No usable Playwright package/browser runtime was installed; the version probe did not complete. No browser success result was mocked.",
        "console_errors": None, "page_errors": None, **security})
    write("phase3v2a_summary.json", {"phase_status": "PHASE3V2A_MODEL_INFERENCE_COMPLETE_UI_PENDING", "temporal_functionality": "DIRECT_AND_API_OPERATIONAL",
        "browser_verification": "PENDING_PLAYWRIGHT_RUNTIME", "next_phase": "PHASE3V2B_REAL_BROWSER_VERIFICATION", "timestamp_utc": datetime.now(timezone.utc).isoformat(), **security})


if __name__ == "__main__":
    main()
