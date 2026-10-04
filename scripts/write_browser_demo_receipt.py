"""Record observations from the real dedicated-control browser audit."""
from __future__ import annotations
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "artifacts" / "runtime_optimization" / "final_verification"
out.mkdir(parents=True, exist_ok=True)
routes = {
    "s2_vqa": {"control": "#run-single-image", "http": 200, "route": "SINGLE_IMAGE_VQA", "status": "COMPLETED", "rendered_result": True, "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings_rendered": True, "provenance_rendered": True},
    "scene": {"control": "#run-single-image", "http": 200, "route": "SINGLE_IMAGE_SCENE_DESCRIPTION", "status": "COMPLETED", "rendered_result": True, "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings_rendered": True, "provenance_rendered": True},
    "sar_vqa": {"control": "#run-sar-vqa", "http": 200, "route": "SINGLE_IMAGE_SAR_VQA", "status": "COMPLETED", "rendered_result": True, "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings_rendered": True, "provenance_rendered": True, "sar_limitation_warning_rendered": True},
    "optical_sar": {"control": "#run-v1", "http": 200, "route": "OPTICAL_SAR_ANALYSIS", "status": "COMPLETED", "rendered_result": True, "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings_rendered": True, "provenance_rendered": True, "coregistration_warning_rendered": True},
    "temporal": {"control": "#run-temporal-change", "http": 200, "route": "TEMPORAL_CHANGE_DESCRIPTION", "status": "COMPLETED", "rendered_result": True, "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings_rendered": True, "provenance_rendered": True, "pre_post_rendered": True, "fake_geometry_claims": []},
}
audit = {"real_browser": "Codex in-app loopback browser", "endpoint": "/api/v1/demo/run", "dedicated_demo_controls": routes, "console_errors": [], "page_errors": [], "failed_requests": [], "uncaught_js_errors": [], "generic_analysis_control_used": False, "source_mode_labels_rendered": True, "stale_result_detected": False, "export_links_created": {"s2_vqa": True, "scene": True, "sar_vqa": True, "optical_sar": True, "temporal": True}}
(out / "browser_demo_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
diagnosis = {"expected_handler": "chooseDemo(kind) then route-specific Run control then runSelectedDemo(kind, question)", "actual_handler": "same expected dedicated handler", "general_form_interference": False, "endpoint_selected": "/api/v1/demo/run", "root_cause": "VERIFICATION_FLOW_ERROR: the earlier check used general #run instead of the selected route-specific control.", "production_frontend_bug": False, "fix_applied": "none"}
(out / "demo_flow_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")
with (out / "run.log").open("a", encoding="utf-8") as log:
    log.write("dedicated browser Demo controls audited; console/page/failed-request counts zero\n")
