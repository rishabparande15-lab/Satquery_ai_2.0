"""Real-browser evidence for the unified agent's admitted temporal workflow.

This script deliberately does not invent S1/S2 samples when their approved
local development source is unavailable.  It exercises the real validation-safe
LEVIR pair through the new unified endpoint instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "agent" / "phase3w"
PAIR = ROOT / "datasets" / "levir_cc" / "development_smoke" / "images" / "val"
CHROME = Path(os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
BASE_URL = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8767")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    requests, responses, console_errors, page_errors, failed = [], [], [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(CHROME), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("requestfailed", lambda request: failed.append({"url": request.url, "failure": str(request.failure)}))
        page.on("request", lambda request: requests.append({"method": request.method, "url": request.url, "post_data": request.post_data}) if "/api/v1/query" in request.url else None)
        page.on("response", lambda response: responses.append({"status": response.status, "url": response.url, "body": response.json()}) if "/api/v1/query" in response.url else None)
        page.goto(BASE_URL, wait_until="networkidle")
        assert page.get_by_role("button", name="Single image").is_visible()
        assert page.get_by_role("button", name="Optical + SAR", exact=True).is_visible()
        page.get_by_role("button", name="Bi-temporal").click()
        assert page.locator("label[for='temporal-t1']").is_visible()
        assert page.locator("label[for='temporal-t2']").is_visible()
        page.locator("#temporal-t1").set_input_files(str(PAIR / "A" / "val_000001.png"))
        page.locator("#temporal-t2").set_input_files(str(PAIR / "B" / "val_000001.png"))
        page.locator("#temporal-question").fill("What changed between these two images?")
        with page.expect_response(lambda response: "/api/v1/query" in response.url and response.request.method == "POST") as awaited:
            page.locator("#run-temporal-change").click()
        response = awaited.value
        assert response.status == 200
        page.wait_for_function("!document.getElementById('run-temporal-change').disabled")
        output = page.locator("#temporal-change-output").inner_text()
        provenance = page.locator("#temporal-change-provenance").inner_text()
        assert "TEMPORAL_CHANGE_DESCRIPTION" in output and "Chg2Cap" in output and "change description:" in output
        assert "PRE_POST" in provenance and "generated_tokens" in provenance
        screenshot = OUT / "satquery-phase3w-temporal-live.png"
        page.screenshot(path=str(screenshot), full_page=True)
        browser.close()
    payload = {"browser": "Google Chrome", "headless": True, "url": BASE_URL, "requests": requests,
               "responses": responses, "output": output, "provenance": provenance,
               "console_errors": console_errors, "page_errors": page_errors, "failed_requests": failed,
               "screenshot": str(screenshot), "test_access": {"levir_content": 0, "levir_labels": 0, "levir_inference": 0, "levir_metrics": 0, "cdvqa": 0}}
    (OUT / "phase3w_browser_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": response.status, "output": output, "console_errors": len(console_errors), "page_errors": len(page_errors), "screenshot": str(screenshot)}, indent=2))


if __name__ == "__main__":
    main()
