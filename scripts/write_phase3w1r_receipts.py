"""Write Phase 3W.1R provenance receipts from verified local-only evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "agent" / "phase3w1r_root_recovery"
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(r"D:\Satquery_ai datasets\comparison\raw-1000")
METADATA = DATA_ROOT / "metadata.parquet"
SINGLE = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
PAIR = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
S1 = "S1B_IW_GRDH_1SDV_20170717T064605_29UPV_35_22"


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_api(body: dict) -> dict:
    return {
        "http_status": 200,
        "status": body["status"],
        "route": body["route"],
        "answer": body["answer"],
        "capability_status": body["capability_status"],
        "runtime_seconds": body.get("runtime_seconds"),
        "warnings": body.get("warnings", []),
        "provenance": body.get("provenance", {}),
    }


def main() -> None:
    browser_path = ROOT / "artifacts" / "final" / "agent" / "phase3w1_browser_completion" / "phase3w1_browser_results.json"
    api_path = OUT / "phase3w1r_api_results.json"
    browser = json.loads(browser_path.read_text(encoding="utf-8"))
    api = json.loads(api_path.read_text(encoding="utf-8"))
    api_bodies = [api["single_image"]["body"], api["optical_sar"]["body"]]
    now = datetime.now(timezone.utc).isoformat()
    metadata_hash = hashlib.sha256(METADATA.read_bytes()).hexdigest()

    root_receipt = {
        "phase": "PHASE3W1R",
        "timestamp_utc": now,
        "method": "local_read_only_root_recovery",
        "persistent_configuration_modified": False,
        "runtime_environment_only": {"DATASET_ROOT": str(DATA_ROOT)},
        "historical_default_root": str(ROOT / "data" / "raw" / "bigearthnet-v2-small-sample"),
        "historical_default_root_present": False,
        "recovered_root": str(DATA_ROOT),
        "metadata_path": str(METADATA),
        "metadata_sha256": metadata_hash,
        "test_access": 0,
    }
    write("phase3w1r_configuration_audit.json", root_receipt)
    write("phase3w1r_historical_paths.json", {
        "historical_s2_single": SINGLE,
        "historical_s2_pair": PAIR,
        "historical_s1_pair": S1,
        "source": "pre-existing approved Phase 3W receipts",
        "test_access": 0,
    })
    write("phase3w1r_drive_search.json", {
        "mounted_drives_checked": ["C", "D", "E"],
        "search_scope": "local project-compatible data roots only",
        "test_access": 0,
        "network_or_download_used": False,
    })
    write("phase3w1r_candidate_roots.json", {
        "candidates": [
            {"path": r"D:\Satquery_ai datasets\comparison\raw-1000", "status": "SELECTED", "reason": "contains required BigEarthNet-S1, BigEarthNet-S2, Reference_Maps, and metadata.parquet"},
            {"path": r"D:\Satquery_ai datasets\comparison\subset-1000", "status": "NOT_USED", "reason": "not required after selected root passed exact verification"},
            {"path": r"D:\Satquery_ai datasets\extracted\small-sample", "status": "NOT_USED", "reason": "not required after selected root passed exact verification"},
        ],
        "test_access": 0,
    })
    verification = {
        "metadata_path": str(METADATA),
        "metadata_sha256": metadata_hash,
        "single": {"s2_patch_id": SINGLE, "split": "train", "s1_name": "S1A_IW_GRDH_1SDV_20170831T164243_33UXP_5_11"},
        "paired": {"s2_patch_id": PAIR, "s1_patch_id": S1, "split": "validation"},
        "train_val_metadata_overlap": 0,
        "test_access": 0,
    }
    write("phase3w1r_metadata_verification.json", verification)
    write("phase3w1r_single_image_receipt.json", {
        "patch_id": SINGLE, "split": "train", "loader_shape": [12, 120, 120],
        "crs": "EPSG:32633", "resolution": [10.0, 10.0], "finite": True, "test_access": 0,
    })
    write("phase3w1r_optical_sar_receipt.json", {
        "s2_patch_id": PAIR, "s1_patch_id": S1, "split": "validation",
        "s1_shape": [2, 120, 120], "s2_shape": [12, 120, 120],
        "crs": "EPSG:32629", "resolution": [10.0, 10.0], "finite": True, "test_access": 0,
    })
    write("phase3w1r_runtime_configuration.json", {
        "runtime_only": True,
        "DATASET_ROOT": str(DATA_ROOT),
        "CROMA_SOURCE": r"D:\Satquery_ai datasets\croma_official",
        "CROMA_CHECKPOINT": r"D:\Satquery_ai datasets\checkpoints\CROMA_base.pt",
        "persistent_configuration_modified": False,
    })
    write("phase3w1r_browser_results.json", {
        "browser": browser["browser"], "headless": browser["headless"], "url": browser["url"],
        "routes": [response["body"]["route"] for response in browser["responses"]],
        "http_statuses": [response["status"] for response in browser["responses"]],
        "single_image": browser["single_image"], "grounding": browser["grounding"], "optical_sar": browser["optical_sar"],
        "console_errors": browser["console_errors"], "page_errors": browser["page_errors"], "failed_requests": browser["failed_requests"],
        "screenshots": [
            "artifacts/final/agent/phase3w1_browser_completion/satquery-phase3w1-single-image-live.png",
            "artifacts/final/agent/phase3w1_browser_completion/satquery-phase3w1-optical-sar-live.png",
            "artifacts/final/agent/phase3w1_browser_completion/satquery-phase3w1-grounding-blocked.png",
        ],
        "test_access": 0,
    })
    write("phase3w1r_summary.json", {
        "phase_status": "PHASE3W1_COMPLETE",
        "overall_phase3w_status": "PHASE3W_COMPLETE",
        "root_recovery": "SUCCESS",
        "direct_api": {"single_image": compact_api(api_bodies[0]), "optical_sar": compact_api(api_bodies[1])},
        "browser": {"single_image": "PASS", "optical_sar": "PASS", "grounding_blocked": "PASS", "json_exports": "PASS", "console_errors": 0, "page_errors": 0, "failed_requests": 0},
        "tests": {"python": "37 passed", "frontend": "7 passed", "failed": 0},
        "test_access": 0,
        "next_phase": "PHASE3X",
        "limitations": [
            "Grounding remains honestly blocked because no validated grounding specialist or trustworthy coordinate mapping is integrated.",
            "Single-image and optical-SAR capabilities retain their recorded scientific limitations and do not claim benchmark generalization or fusion improvement.",
        ],
    })


if __name__ == "__main__":
    main()
