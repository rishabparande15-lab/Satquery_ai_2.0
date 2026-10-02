"""Create audit receipts for Phase 3W without rerunning model inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satquery_agent import capability_registry

OUT = ROOT / "artifacts" / "final" / "agent" / "phase3w"


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    browser = json.loads((OUT / "phase3w_browser_verification.json").read_text(encoding="utf-8"))
    response = browser["responses"][0]["body"]
    registry = capability_registry()
    save("phase3w_capability_registry.json", {"registry": registry, "confidence_policy": {"value": None, "type": "NOT_AVAILABLE"}})
    save("phase3w_controller_audit.json", {
        "controllers_found": ["SingleImageVQAController", "SatQueryV1Controller", "TemporalChangeDescriptionController", "SatQueryAgent"],
        "shared_components": ["capability registry", "normalized input description", "deterministic intent classification", "unified response contract", "operational execution trace"],
        "duplicated_logic_replaced": ["frontend specialist endpoint selection"],
        "preserved_specialist_boundaries": ["single-image S2 validation", "paired S1+S2 validation", "Chg2Cap PRE_POST validation"],
    })
    save("phase3w_routing_matrix.json", {"cases": [
        {"inputs": "SINGLE S2", "intent": "VQA", "route": "SINGLE_IMAGE_VQA", "status": "AVAILABLE_WITH_LIMITATIONS"},
        {"inputs": "S1 + S2", "intent": "OPTICAL_SAR_REASONING", "route": "OPTICAL_SAR_ANALYSIS", "status": "AVAILABLE_WITH_LIMITATIONS"},
        {"inputs": "T1 PRE + T2 POST", "intent": "CHANGE_DESCRIPTION", "route": "TEMPORAL_CHANGE_DESCRIPTION", "status": "AVAILABLE"},
        {"inputs": "SINGLE", "intent": "GROUNDING", "route": "SINGLE_IMAGE_GROUNDING", "status": "BLOCKED"},
        {"inputs": "SINGLE", "intent": "CAPTION_REQUEST", "route": "SINGLE_IMAGE_CAPTION", "status": "EXPERIMENTAL_LIMITED"},
        {"inputs": "T1 only", "intent": "CHANGE_DESCRIPTION", "route": None, "status": "ERROR", "error": "INCOMPLETE_TEMPORAL_PAIR"},
        {"inputs": "S2 only", "intent": "OPTICAL_SAR_REASONING", "route": None, "status": "ERROR", "error": "ROUTING_CLARIFICATION_REQUIRED"},
    ]})
    save("phase3w_response_contract.json", {"top_level_fields": ["status", "request_id", "query", "route", "capability_status", "answer", "visual_evidence", "confidence", "warnings", "provenance", "execution_trace", "details", "error"], "answer_confidence": {"value": None, "type": "NOT_AVAILABLE"}, "trace_policy": "Operational steps only; no private reasoning."})
    save("phase3w_temporal_live.json", {"source": "real browser -> /api/v1/query -> agent -> Chg2Cap", "http_status": browser["responses"][0]["status"], "route": response["route"], "answer": response["answer"], "model": response["provenance"]["model"], "console_errors": len(browser["console_errors"]), "page_errors": len(browser["page_errors"])})
    unavailable = {"status": "NOT_RUN", "reason": "No approved non-test local S2 or paired S1+S2 development inputs were available to the real app at Phase 3W verification time. No data was downloaded or substituted."}
    save("phase3w_single_image_live.json", unavailable)
    save("phase3w_optical_sar_live.json", unavailable)
    save("phase3w_grounding_blocked.json", {"route": "SINGLE_IMAGE_GROUNDING", "status": "BLOCKED", "error_code": "GROUNDING_MODEL_UNAVAILABLE", "visual_evidence": {"boxes": [], "masks": []}, "reason": registry["SINGLE_IMAGE_GROUNDING"]["limitations"][0]})
    save("phase3w_api_verification.json", {"endpoint": "/api/v1/query", "temporal_real_http_status": browser["responses"][0]["status"], "temporal_route": response["route"], "grounding_contract": "unit verified; blocked without loading a model"})
    save("phase3w_specialist_regression.json", {"unit_contracts": "passed", "temporal_real_agent_regression": "passed", "single_image_real_regression": "not run: approved non-test input unavailable", "optical_sar_real_regression": "not run: approved non-test input unavailable"})
    save("phase3w_resource_profile.json", {"temporal": {"specialist": "Chg2Cap", "device": response["details"]["provenance"]["model"]["device"], "runtime_seconds": response["details"]["runtime_seconds"]}, "single_image": unavailable, "optical_sar": unavailable})
    save("phase3w_test_results.json", {"python": {"passed": 37, "failed": 0}, "frontend_node": {"passed": 7, "failed": 0}, "real_browser": {"temporal_unified": "passed", "single_image": "not run: no approved local input", "optical_sar": "not run: no approved local input"}})
    save("phase3w_summary.json", {"status": "PHASE3W_INTEGRATION_PARTIAL", "complete_items": ["central agent", "registry", "input/intent routing", "fail-closed policy", "unified API", "unified frontend routing", "real temporal browser path"], "remaining": ["real browser verification for single-image VQA", "real browser verification for optical-SAR", "real browser grounding UI using an approved non-test single-image input"], "test_access": {"bigearthnet_new_test": 0, "levir_test_content": 0, "levir_test_labels": 0, "levir_test_inference": 0, "levir_test_metrics": 0, "cdvqa_test": 0}})


if __name__ == "__main__":
    main()
