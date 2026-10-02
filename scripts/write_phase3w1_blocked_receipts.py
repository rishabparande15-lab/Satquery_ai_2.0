"""Persist the approved-input recovery result without accessing new data."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "agent" / "phase3w1_browser_completion"

def save(name: str, data: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def main() -> None:
    absent_root = str(ROOT / "data" / "raw" / "bigearthnet-v2-small-sample")
    single_id = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
    pair_id = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
    reason = "Configured local BigEarthNet root is absent; metadata.parquet and historical raster files are not present in this workspace."
    save("phase3w1_input_recovery.json", {"status": "PHASE3W1_APPROVED_INPUTS_NOT_FOUND", "checked_sources": ["artifacts/final/single_image/phase3u/phase3u_frontend_verification.json", "artifacts/final/single_image/phase3u/phase3u_vqa_results.json", "artifacts/final/satquery_v1/satquery_v1_multimodal_e2e_results.json", "artifacts/final/satquery_v1/satquery_v1_multimodal_validation_manifest.json", absent_root], "dataset_root": absent_root, "dataset_root_exists": False, "missing": ["metadata.parquet", "approved non-test S2 raster stack", "approved non-test paired S1/S2 raster stack"], "test_directories_opened": False, "data_downloaded": False})
    save("phase3w1_single_image_input_receipt.json", {"found": False, "historical_sample_id": single_id, "historical_split": "train", "historical_evidence": "Phase 3U real-browser receipt", "physical_path": None, "validation": "NOT_RUN: " + reason, "test_access": 0})
    save("phase3w1_optical_sar_input_receipt.json", {"found": False, "historical_pair_id": pair_id, "historical_split": "non-TEST fixed panel", "historical_evidence": "SatQuery v1 multimodal E2E receipt", "s1_path": None, "s2_path": None, "validation": "NOT_RUN: " + reason, "test_access": 0})
    not_run = {"status": "NOT_RUN", "reason": reason, "endpoint": "/api/v1/query", "no_fallback": True}
    save("phase3w1_single_image_api.json", not_run)
    save("phase3w1_optical_sar_api.json", not_run)
    save("phase3w1_single_image_browser.json", not_run)
    save("phase3w1_optical_sar_browser.json", not_run)
    save("phase3w1_grounding_blocked_browser.json", {**not_run, "route": "SINGLE_IMAGE_GROUNDING", "reason": "Requires the same approved S2 input; no standalone test image was substituted."})
    save("phase3w1_routing_isolation.json", {"status": "UNIT_AND_PRIOR_TEMPORAL_RECEIPT_ONLY", "single_image": "not browser-run", "optical_sar": "not browser-run", "temporal": "previous Phase 3W real-browser receipt passed", "grounding": "unit contract remains fail-closed"})
    save("phase3w1_export_verification.json", {"status": "IMPLEMENTED_NOT_BROWSER_EXERCISED", "mechanism": "client-side JSON download from unified result panels", "reason": "No approved local successful single-image or paired result could be generated in this workspace."})
    save("phase3w1_test_results.json", {"new_model_or_browser_runs": 0, "reason": reason, "existing_phase3w_tests": {"python_passed": 37, "frontend_passed": 7}})
    save("phase3w1_summary.json", {"phase_status": "PHASE3W1_APPROVED_INPUTS_NOT_FOUND", "overall_phase3w_status": "PHASE3W_INTEGRATION_PARTIAL", "blocker": reason, "historical_ids_recovered": {"single_image": single_id, "optical_sar": pair_id}, "test_access": {"bigearthnet_new_test": 0, "levir_test_content": 0, "levir_test_labels": 0, "levir_test_inference": 0, "levir_test_metrics": 0, "cdvqa_test": 0}, "next_step": "Restore or explicitly authorize the exact approved non-test local BigEarthNet development root; do not download or substitute data automatically."})

if __name__ == "__main__":
    main()
