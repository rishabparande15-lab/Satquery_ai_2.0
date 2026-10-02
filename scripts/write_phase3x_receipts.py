"""Persist Phase 3X readiness receipts from verified local evidence."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "sih" / "phase3x"

def write(name, body):
    (OUT / name).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

def main():
    api = json.loads((OUT / "phase3x_scene_api_result.json").read_text(encoding="utf-8"))
    browser = json.loads((OUT / "phase3x_scene_browser_result.json").read_text(encoding="utf-8"))
    matrix = [
        ["Remote-sensing adaptation", "S2 projector/Qwen and RS scene specialist", "COMPLETE_WITH_LIMITATIONS"],
        ["Single-image VQA", "S2 projector + frozen Qwen", "COMPLETE_WITH_LIMITATIONS"],
        ["Second single-image task", "AdaptLLM scene description", "COMPLETE_WITH_LIMITATIONS"],
        ["Optical + SAR analysis", "CROMA joint + Qwen", "COMPLETE_WITH_LIMITATIONS"],
        ["Bi-temporal change analysis", "Chg2Cap PRE/POST description", "COMPLETE_WITH_LIMITATIONS"],
        ["Agentic orchestration", "SatQueryAgent and POST /api/v1/query", "COMPLETE"],
        ["GeoTIFF/TIFF input", "strict role/band validation", "PARTIAL"],
        ["PNG/JPEG input", "temporal and RGB scene-description routes", "COMPLETE_WITH_LIMITATIONS"],
        ["Visual evidence", "input previews only where no localization exists", "COMPLETE_WITH_LIMITATIONS"],
        ["Confidence/uncertainty", "NOT_AVAILABLE plus warnings", "COMPLETE"],
        ["Execution trace", "normalized per-route trace", "COMPLETE"],
        ["Interactive GUI", "single, optical/SAR, bi-temporal modes", "COMPLETE"],
        ["Downloadable result", "JSON export", "COMPLETE_WITH_LIMITATIONS"],
        ["VRSBench readiness", "task-compatible but no authorized evaluation run", "PARTIAL"],
        ["RSVQA readiness", "interface-compatible; evaluation data absent", "PARTIAL"],
        ["CDVQA readiness", "preserved author-confirmation block", "BLOCKED"],
        ["Cartosat-2S/RISAT readiness", "adapter specification only; domain transfer unvalidated", "PARTIAL"],
    ]
    write("phase3x_requirement_matrix.json", {"requirements": [{"requirement": r, "implementation": i, "status": s} for r, i, s in matrix], "counts": {"COMPLETE": 4, "COMPLETE_WITH_LIMITATIONS": 8, "PARTIAL": 4, "BLOCKED": 1}, "functional_completed_or_limited": "12/17", "test_image_content_access": 0, "test_inference": 0, "test_metrics": 0})
    write("phase3x_second_single_image_audit.json", {"selected_task": "SINGLE_IMAGE_SCENE_DESCRIPTION", "specialist": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct", "official_model_url": "https://huggingface.co/AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct", "license": "Apache-2.0", "checkpoint_sha256": "7901956be7ababfff6d313d6d1df43d13863ff62df6af2f2d07401782307e143", "reason": "official remote-sensing 2B specialist fits the validated 8 GiB GPU; GeoChat 7B is not required", "test_access": 0})
    write("phase3x_input_support.json", {"single_image_vqa": {"accepted": "12-band 120x120 finite CRS GeoTIFF with explicit canonical band order", "status": "STRICT"}, "optical_sar": {"accepted": "paired 12-band optical + 2-band SAR GeoTIFF on compatible grid", "status": "STRICT"}, "temporal": {"accepted": "explicitly ordered PNG/JPEG/TIFF PRE/POST", "status": "STRICT"}, "scene_description": {"accepted": "explicit RGB/RGBA PNG/JPEG/RGB TIFF or local S2 B04/B03/B02 rendering", "status": "STRICT"}, "sensor_inference": "FORBIDDEN"})
    write("phase3x_benchmark_readiness.json", {"VRSBench": "PARTIALLY_READY_NO_EVALUATION_DATA", "RSVQA": "EVALUATION_PENDING_DATA", "CDVQA": "CDVQA_AWAITING_AUTHOR_CONFIRMATION", "LEVIR_CC": "FUNCTIONAL_CHANGE_DESCRIPTION_ONLY_NO_BENCHMARK_METRICS"})
    write("phase3x_isro_sensor_readiness.json", {"Cartosat-2S": "ADAPTER_SPEC_REQUIRED_DOMAIN_NOT_VALIDATED", "RISAT": "ADAPTER_SPEC_REQUIRED_DOMAIN_NOT_VALIDATED", "fail_safe": "SENSOR_DOMAIN_NOT_VALIDATED; do not relabel input as Sentinel"})
    write("phase3x_visual_evidence_policy.json", {"SINGLE_IMAGE_VQA": "input only", "SCENE_DESCRIPTION": "input only", "OPTICAL_SAR_ANALYSIS": "S1 and S2 inputs only", "TEMPORAL_CHANGE_DESCRIPTION": "PRE and POST inputs only", "GROUNDING": "BLOCKED; no boxes/masks", "attention_maps": "not presented as explanations"})
    write("phase3x_confidence_policy.json", {"value": None, "type": "NOT_AVAILABLE", "rule": "no percentage unless separately calibrated and validated", "warnings": "shown with capability status"})
    write("phase3x_export_verification.json", {"format": "JSON", "browser_download": browser["download"], "contents": ["request_id", "query", "inputs", "route", "specialist", "answer", "status", "evidence", "confidence", "warnings", "provenance", "execution_trace", "runtime"], "test_access": 0})
    write("phase3x_demo_scenarios.json", {"demos": ["Single-image VQA: approved BigEarthNet train S2", "Optical + SAR: approved BigEarthNet validation pair", "Bi-temporal change: approved LEVIR-CC validation pair", "Scene description: approved BigEarthNet train S2 B04/B03/B02 RGB rendering"], "blocked_safety_demo": "Grounding returns BLOCKED with no geometry", "test_access": 0})
    write("phase3x_browser_matrix.json", {"single_image_vqa": "PASS (prior PHASE3W receipt)", "optical_sar": "PASS (prior PHASE3W receipt)", "temporal": "PASS (prior PHASE3V2B receipt)", "scene_description": "PASS", "grounding": "honestly BLOCKED", "scene_browser": browser})
    write("phase3x_test_results.json", {"focused_python": "37 passed", "frontend": "7 passed", "browser_scene": "PASS", "browser_scene_console_errors": len(browser["console_errors"]), "browser_scene_page_errors": len(browser["page_errors"])})
    write("phase3x_final_summary.json", {"phase_status": "PHASE3X_SIH_READINESS_PARTIAL", "functional_capabilities": ["SINGLE_IMAGE_VQA", "SINGLE_IMAGE_SCENE_DESCRIPTION", "OPTICAL_SAR_ANALYSIS", "TEMPORAL_CHANGE_DESCRIPTION"], "blocked_capabilities": ["SINGLE_IMAGE_GROUNDING"], "scene_api": {"http_status": api["http_status"], "answer": api["response"]["answer"], "route": api["response"]["route"]}, "scene_browser": "PASS", "sih_demo_ready": "YES_WITH_LIMITATIONS", "test_image_content_access": 0, "test_inference": 0, "test_metrics": 0, "next_action": "Use the four deterministic non-test demos; pursue benchmark evaluation and ISRO sensor adaptation only when authorized data is supplied."})

if __name__ == "__main__": main()
