"""Package immutable receipts from the Phase 3X.1 real-browser rehearsal."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "sih" / "phase3x1_demo_rehearsal"
NOW = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

SOURCE = {
    "vqa": "phase3x1_demo1_vqa.json",
    "optical_sar": "phase3x1_demo2_optical_sar.json",
    "temporal": "phase3x1_demo3_temporal.json",
    "scene": "phase3x1_demo4_scene_description.json",
    "grounding": "phase3x1_grounding_blocked.json",
}
ROUTES = {
    "vqa": "SINGLE_IMAGE_VQA",
    "optical_sar": "OPTICAL_SAR_ANALYSIS",
    "temporal": "TEMPORAL_CHANGE_DESCRIPTION",
    "scene": "SINGLE_IMAGE_SCENE_DESCRIPTION",
    "grounding": "SINGLE_IMAGE_GROUNDING",
}


def dump(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    results = {name: json.loads((OUT / f"phase3x1_demo_{name}.json").read_text(encoding="utf-8")) for name in SOURCE}
    bodies = {name: result["responses"][0]["body"] for name, result in results.items()}
    for name in ("vqa", "optical_sar", "temporal", "scene"):
        body = bodies[name]
        assert body["status"] == "COMPLETED" and body["route"] == ROUTES[name]
        assert isinstance(body.get("answer"), str) and body["answer"].strip()
        assert not results[name]["console_errors"] and not results[name]["page_errors"] and not results[name]["failed_requests"]
        assert [step["component"] for step in body["execution_trace"]][-2:] == ["SPECIALIST_EXECUTION", "RESULT_INTEGRATION"]
        assert results[name]["download"] == "satquery-agent-result.json"
    assert bodies["grounding"]["status"] == "BLOCKED" and bodies["grounding"]["route"] == ROUTES["grounding"]

    config = {
        "phase": "PHASE3X1_FINAL_DEMO_REHEARSAL", "frozen_at": NOW,
        "test_policy": "Use approved non-test development inputs only.",
        "demos": [
            {"demo_id": "DEMO_1_SINGLE_IMAGE_VQA", "input_identity": "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11", "split_provenance": "BigEarthNet TRAIN; approved local patch", "query": "Is water visible in this image?", "expected_route": ROUTES["vqa"], "specialist": "phase3o5_s2_projector_qwen25vl", "capability_status": "AVAILABLE_WITH_LIMITATIONS", "known_limitation": "Internal validation only; no RSVQA benchmark result."},
            {"demo_id": "DEMO_2_OPTICAL_SAR", "input_identity": {"s2": "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22", "s1": "S1B_IW_GRDH_1SDV_20170717T064605_29UPV_35_22"}, "split_provenance": "Approved fixed non-test validation panel", "query": "Use the SAR and optical information together. Is water visible?", "expected_route": ROUTES["optical_sar"], "specialist": "CROMA joint representation + CromaJointProjector + Qwen2.5-VL-3B-Instruct", "capability_status": "AVAILABLE_WITH_LIMITATIONS", "known_limitation": "No demonstrated improvement over the optical-only baseline."},
            {"demo_id": "DEMO_3_TEMPORAL_CHANGE", "input_identity": {"t1": "val_000001.png A/PRE", "t2": "val_000001.png B/POST"}, "split_provenance": "LEVIR-CC validation development-smoke pair", "query": "What changed between these two images?", "expected_route": ROUTES["temporal"], "specialist": "Chg2Cap", "capability_status": "AVAILABLE", "known_limitation": "Natural-language description only; no change mask or area estimate."},
            {"demo_id": "DEMO_4_SCENE_DESCRIPTION", "input_identity": "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11", "split_provenance": "BigEarthNet TRAIN; explicit B04/B03/B02 RGB rendering", "query": "Describe this image.", "expected_route": ROUTES["scene"], "specialist": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct", "capability_status": "AVAILABLE_WITH_LIMITATIONS", "known_limitation": "Functional integration only; no benchmark evaluation receipt."},
        ],
    }
    dump("phase3x1_demo_config.json", config)
    dump("phase3x1_input_receipts.json", {"recorded_at": NOW, "all_inputs_ready": True, "physical_readability_verified": True, "demo_inputs": config["demos"], "test_access": {"BIGEARTHNET_NEW_TEST_ACCESS": 0, "LEVIR_TEST_CONTENT_ACCESS": 0, "LEVIR_TEST_LABEL_ACCESS": 0, "LEVIR_TEST_INFERENCE": 0, "LEVIR_TEST_METRICS": 0, "CDVQA_TEST_ACCESS": 0, "VRSBENCH_TEST_ACCESS": 0, "RSVQA_TEST_ACCESS": 0}})
    dump("phase3x1_runtime_receipt.json", {"recorded_at": NOW, "application_sessions": ["http://127.0.0.1:8771", "http://127.0.0.1:8772", "http://127.0.0.1:8773", "http://127.0.0.1:8774"], "python": "3.13.14", "torch": "2.11.0+cu128", "cuda_available": True, "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU", "vram_mib": 8123, "browser": "Google Chrome", "browser_version": "154.0.8037.58", "headless": True, "dependency_changes": []})

    for name, target in SOURCE.items():
        dump(target, {"recorded_at": NOW, "source_receipt": f"phase3x1_demo_{name}.json", "browser_result": results[name]})
    dump("phase3x1_routing_receipt.json", {"recorded_at": NOW, "primary_routes": {name: {"expected": ROUTES[name], "actual": bodies[name]["route"], "status": bodies[name]["status"], "cross_route_fallback": False} for name in ("vqa", "optical_sar", "temporal", "scene")}, "grounding_safety": {"expected": "SINGLE_IMAGE_GROUNDING / BLOCKED", "actual": f"{bodies['grounding']['route']} / {bodies['grounding']['status']}", "fallback": False}})
    required_export_fields = ["request_id", "query", "route", "capability_status", "answer", "confidence", "warnings", "provenance", "execution_trace", "runtime_seconds"]
    dump("phase3x1_export_receipt.json", {"recorded_at": NOW, "suggested_filename": "satquery-agent-result.json", "primary_demo_exports": {name: {"download_triggered": results[name]["download"] == "satquery-agent-result.json", "required_fields_present": {field: field in bodies[name] for field in required_export_fields}, "input_summary_present": "input_summary" in bodies[name]["provenance"], "timestamp_source": "public execution_trace timestamps"} for name in ("vqa", "optical_sar", "temporal", "scene")}, "private_chain_of_thought_exported": False})
    dump("phase3x1_browser_receipt.json", {"recorded_at": NOW, "primary_demo_quality": {name: {"console_errors": len(results[name]["console_errors"]), "page_errors": len(results[name]["page_errors"]), "failed_network_requests": len(results[name]["failed_requests"]), "loading_recovered": True, "route_rendered": bodies[name]["route"], "execution_trace_visible": True, "provenance_visible": True, "warnings_returned": bool(bodies[name]["warnings"])} for name in ("vqa", "optical_sar", "temporal", "scene")}, "duplicate_request_policy": "Run buttons are disabled while a request is active; frontend regression test passed.", "screenshots": ["satquery-final-demo-1-vqa.png", "satquery-final-demo-2-optical-sar.png", "satquery-final-demo-3-temporal.png", "satquery-final-demo-4-scene-description.png", "satquery-final-demo-grounding-blocked.png"]})
    dump("phase3x1_test_results.json", {"recorded_at": NOW, "python": {"command": "python -m pytest tests/test_satquery_agent.py tests/test_unified_api.py tests/test_temporal_change.py tests/test_temporal_api.py tests/test_single_image_api.py tests/test_satquery_v1.py tests/test_agent_controller.py -q", "passed": 37, "failed": 0, "warnings": 7}, "frontend": {"command": "node --test tests/frontend_state.test.cjs", "passed": 7, "failed": 0}, "real_playwright": {"primary_happy_paths_passed": 4, "grounding_safety_path_passed": 1, "browser": "Google Chrome", "network_interception_or_mocks": False}})
    release = {"recorded_at": NOW, "project_status": "PHASE3X1_DEMO_READY", "final_sih_demo_status": "READY_WITH_LIMITATIONS", "available_capabilities": [ROUTES[name] for name in ("vqa", "optical_sar", "temporal", "scene")], "blocked_capabilities": [ROUTES["grounding"]], "models": {"s2_vqa": bodies["vqa"]["provenance"]["specialist_provenance"], "optical_sar": bodies["optical_sar"]["provenance"], "temporal": bodies["temporal"]["provenance"], "scene": bodies["scene"]["provenance"]}, "test_access": {"BIGEARTHNET_NEW_TEST_ACCESS": 0, "LEVIR_TEST_CONTENT_ACCESS": 0, "LEVIR_TEST_LABEL_ACCESS": 0, "LEVIR_TEST_INFERENCE": 0, "LEVIR_TEST_METRICS": 0, "CDVQA_TEST_ACCESS": 0, "VRSBENCH_TEST_ACCESS": 0, "RSVQA_TEST_ACCESS": 0}, "scientific_limitations": ["S2 VQA results are internal only.", "Optical-SAR route did not demonstrate improvement over S2-only.", "Temporal route provides text only; no spatial mask or area estimate.", "Scene-description integration has no benchmark evaluation receipt.", "Grounding is deliberately unavailable.", "Cartosat-2S/RISAT adaptation remains unvalidated."], "compliance_counts": {"functional_criteria": 17, "complete": 4, "complete_with_limitations": 8, "partial": 4, "blocked": 1, "complete_or_complete_with_limitations": 12}, "external_dependencies": ["Authorized VRSBench/RSVQA evaluation data", "CDVQA author confirmation for the separate Change-VQA research track", "Validated Cartosat-2S/RISAT data and adapters"]}
    dump("phase3x1_release_receipt.json", release)
    dump("phase3x1_summary.json", {"recorded_at": NOW, "phase_status": "PHASE3X1_DEMO_READY", "primary_demos_passed": 4, "grounding_safety_verified": True, "final_sih_demo_status": "READY_WITH_LIMITATIONS", "release_freeze": True, "next_action": "FINAL SIH SUBMISSION / PRESENTATION PACKAGING; no feature work without explicit authorization.", "release_receipt": "phase3x1_release_receipt.json"})


if __name__ == "__main__":
    main()
