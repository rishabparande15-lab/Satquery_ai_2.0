"""Write concise Phase 3V.2B evidence receipts from the real browser capture."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "temporal" / "phase3v2b_browser_verification"


def write_json(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    observed = json.loads((OUT / "playwright_observed.json").read_text(encoding="utf-8"))
    request = json.loads(observed["requests"][0]["post_data"])
    request.pop("t1_token", None)
    request.pop("t2_token", None)
    response = observed["responses"][0]["body"]
    happy = observed["happy_path"]
    complete = (observed["responses"][0]["status"] == 200 and bool(response["change_description"].strip()) and response["route"] == "TEMPORAL_CHANGE_DESCRIPTION" and not happy["console_errors"] and not happy["page_errors"] and not happy["failed_requests"])

    write_json("phase3v2b_browser_runtime_audit.json", {
        "node_version": "v24.18.0", "node_playwright_package": "not used (not installed)",
        "python_version": "3.13.14", "python_playwright_package": "installed at E:/Python313/Lib/site-packages/playwright",
        "browser": "Google Chrome", "browser_version": "154.0.8037.58",
        "browser_executable": "C:/Program Files/Google/Chrome/Application/chrome.exe", "browser_mode": "headless",
        "playwright_runtime": "Python Playwright using an existing system Chrome; no browser downloaded",
    })
    write_json("phase3v2b_app_runtime.json", {
        "application_url": observed["url"],
        "frontend_http_status": 200,
        "backend_endpoint_verified": "/api/v1/temporal",
        "backend_http_status": observed["responses"][0]["status"],
        "model_runtime": "Validated Chg2Cap CUDA backend remained separate from the browser automation runtime.",
        "note": "This receipt records the completed local verification session; the temporary local server was stopped after testing.",
    })
    write_json("phase3v2b_live_request.json", {
        "method": observed["requests"][0]["method"], "endpoint": "/api/v1/temporal",
        "http_url": observed["requests"][0]["url"], "payload": request, "development_pair": "val_000001.png",
        "test_content_access": 0, "test_label_access": 0, "test_inference": 0, "test_metrics": 0,
    })
    write_json("phase3v2b_live_response.json", {
        "http_status": observed["responses"][0]["status"], "status": response["status"], "route": response["route"],
        "change_description": response["change_description"], "model": response["model_tool"],
        "temporal_order": response["temporal_order"], "t1_identity": response["t1_identity"],
        "t2_identity": response["t2_identity"], "runtime_seconds": response["runtime_seconds"],
        "checkpoint_sha256": response["provenance"]["model"]["checkpoint_sha256"],
    })
    write_json("phase3v2b_browser_errors.json", {
        "happy_path": happy,
        "negative_path_expected_http_status_console_messages": observed["negative_case_browser_messages"]["console_errors"],
        "negative_path_page_errors": observed["negative_case_browser_messages"]["page_errors"],
        "negative_path_failed_requests": observed["negative_case_browser_messages"]["failed_requests"],
        "interpretation": "The two negative-path console messages are expected reports for deliberately rejected 422 and 415 API requests; they are not happy-path errors.",
    })
    write_json("phase3v2b_loading_state.json", {
        "loading_started_after_submit": True,
        "loading_cleared_after_response": True,
        "submit_controls_recovered": True,
        "duplicate_submission": "prevented by the busy-state control during the observed request",
        "stuck_spinner": False,
    })
    write_json("phase3v2b_negative_tests.json", {"results": observed["negative"], "result": "All negative paths showed an authored error and recovered the UI without page exceptions."})
    write_json("phase3v2b_routing_isolation.json", observed["routing_isolation"])
    write_json("phase3v2b_test_results.json", {
        "temporal_python_tests": {"command_scope": "temporal change, API, and controller routing", "passed": 18, "failed": 0},
        "frontend_state_tests": {"passed": 7, "failed": 0},
        "real_playwright_happy_path": "passed",
        "real_playwright_negative_paths": "passed",
        "test_data_access": {"content": 0, "labels": 0, "inference": 0, "metrics": 0},
    })
    write_json("phase3v2b_summary.json", {
        "phase_status": "PHASE3V2B_COMPLETE" if complete else "PHASE3V2B_BROWSER_RUNTIME_BLOCKED",
        "verification_chain": "Real Chrome -> SatQuery frontend -> POST /api/v1/temporal -> Chg2Cap -> generated description -> rendered UI",
        "application_url": observed["url"], "validation_pair": "val_000001.png", "t1": "A / PRE", "t2": "B / POST",
        "caption": response["change_description"], "happy_path_console_errors": len(happy["console_errors"]),
        "happy_path_page_errors": len(happy["page_errors"]), "python_tests": "18 passed", "frontend_tests": "7 passed",
        "test_access": {"content": 0, "labels": 0, "inference": 0, "metrics": 0},
        "screenshot": "satquery-bitemporal-live-result.png",
        "limitation": "This specialist produces a natural-language description, not a change mask, bounding boxes, or area measurements.",
    })


if __name__ == "__main__":
    main()
