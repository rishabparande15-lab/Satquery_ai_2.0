"""One-route Chrome verification for Phase 3AC.4 public provenance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "runtime" / "phase3ac4"
BASE = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8000")
PATCH = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
CHROME = os.environ.get(
    "SATQUERY_BROWSER_EXECUTABLE",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
_LOCAL_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/]|(?:huggingface[\\/](?:hub|cache))|models--[^\\/\s]+[\\/](?:snapshots|refs)[\\/])"
)


def get_json(path: str) -> dict:
    with urlopen(BASE + path, timeout=60) as response:
        return json.loads(response.read())


def find_local_paths(value, key="$"):
    found = []
    if isinstance(value, dict):
        for name, item in value.items():
            found.extend(find_local_paths(item, f"{key}.{name}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_local_paths(item, f"{key}[{index}]"))
    elif isinstance(value, str) and _LOCAL_PATH.search(value):
        found.append(key)
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    console_errors, page_errors, failed_requests = [], [], []
    health_before = get_json("/api/v1/health")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("requestfailed", lambda request: failed_requests.append({"url": request.url, "failure": str(request.failure)}))
        page.goto(BASE, wait_until="networkidle")
        page.get_by_role("button", name="Optical + SAR", exact=True).click()
        page.locator("#v1-patch-id").evaluate(
            "(select, value) => { if (![...select.options].some(option => option.value === value)) select.add(new Option(value, value)); select.value = value; }",
            PATCH,
        )
        page.locator("#v1-task-type").select_option("binary_qa")
        page.locator("#v1-question").fill("Use the SAR and optical information together. Is water visible?")
        with page.expect_response(
            lambda response: "/api/v1/query" in response.url and response.request.method == "POST",
            timeout=240_000,
        ) as response_info:
            page.locator("#run-v1").click()
        response = response_info.value
        result = response.json()
        assert response.status == 200, result
        assert result.get("route") == "OPTICAL_SAR_ANALYSIS", result
        assert result.get("answer") == "Yes", result
        assert result.get("evidence", {}).get("pair", {}).get("coregistration") == "COREGISTRATION_NOT_VERIFIED", result
        assert result.get("confidence") == {"value": None, "type": "NOT_AVAILABLE"}, result
        assert result.get("details", {}).get("model_id") == MODEL_ID, result
        assert result.get("details", {}).get("qwen_revision") == REVISION, result
        assert result.get("provenance", {}).get("specialist_provenance", {}).get("joint_projector_sha256"), result
        assert not find_local_paths(result), find_local_paths(result)

        page.wait_for_function("document.querySelector('#v1-export') && !document.querySelector('#v1-export').hidden")
        evidence_text = page.locator("#v1-evidence").inner_text()
        provenance_text = page.locator("#v1-provenance").inner_text()
        assert "IMAGE STATISTICS" in evidence_text and "COREGISTRATION_NOT_VERIFIED" in evidence_text
        assert MODEL_ID in provenance_text and REVISION in provenance_text
        assert page.locator("#v1-provenance-details").evaluate("element => element.open")

        with page.expect_download() as download_info:
            page.locator("#v1-export").click()
        export_path = OUT / "optical_sar_export.json"
        download_info.value.save_as(str(export_path))
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        assert exported == result
        assert not find_local_paths(exported), find_local_paths(exported)
        page.screenshot(path=str(OUT / "optical_sar_browser.png"), full_page=True)
        browser.close()

    memory = get_json("/api/v1/runtime/memory")
    health_after = get_json("/api/v1/health")
    ready = get_json("/api/v1/ready")
    receipt = {
        "http_status": response.status,
        "route": result["route"],
        "answer": result["answer"],
        "coregistration_warning": result["evidence"]["pair"]["coregistration"],
        "confidence": result["confidence"],
        "evidence_rendered": "IMAGE STATISTICS" in evidence_text,
        "provenance_rendered": MODEL_ID in provenance_text and REVISION in provenance_text,
        "json_export": str(export_path.relative_to(ROOT)),
        "local_path_leaks": find_local_paths(exported),
        "model_id": result["details"]["model_id"],
        "qwen_revision": result["details"]["qwen_revision"],
        "joint_projector_sha256": result["provenance"]["specialist_provenance"]["joint_projector_sha256"],
        "console_errors": console_errors,
        "page_errors": page_errors,
        "failed_requests": failed_requests,
        "health_before": health_before,
        "health_after": health_after,
        "ready": ready,
        "gpu_memory": memory,
    }
    (OUT / "optical_sar_browser.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "http_status", "route", "answer", "coregistration_warning", "evidence_rendered",
        "provenance_rendered", "local_path_leaks", "console_errors", "page_errors", "failed_requests",
    )}, indent=2))


if __name__ == "__main__":
    main()