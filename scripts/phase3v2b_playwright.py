"""Real-browser verifier for the TEST-safe Phase 3V.2B happy path."""
from __future__ import annotations

import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "temporal" / "phase3v2b_browser_verification"
PAIR = ROOT / "datasets" / "levir_cc" / "development_smoke" / "images" / "val"
CHROME = Path(os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
BASE_URL = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8766")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    console_errors, page_errors, failed, requests, responses = [], [], [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(CHROME), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("requestfailed", lambda request: failed.append({"url": request.url, "failure": str(request.failure)}))
        page.on("request", lambda request: requests.append({"method": request.method, "url": request.url, "post_data": request.post_data}) if "/api/v1/temporal" in request.url else None)
        page.on("response", lambda response: responses.append({"status": response.status, "url": response.url, "body": response.json()}) if "/api/v1/temporal" in response.url else None)
        page.goto(BASE_URL, wait_until="networkidle")
        page.get_by_role("button", name="Bi-temporal").click()
        assert page.locator("#temporal-change-panel").is_visible()
        assert page.locator("label[for='temporal-t1']").is_visible()
        assert page.locator("label[for='temporal-t2']").is_visible()
        assert not page.locator("#paired-v1-panel").is_visible()
        page.locator("#temporal-t1").set_input_files(str(PAIR / "A" / "val_000001.png"))
        page.locator("#temporal-t2").set_input_files(str(PAIR / "B" / "val_000001.png"))
        assert page.locator("#temporal-t1-preview").is_visible()
        assert page.locator("#temporal-t2-preview").is_visible()
        page.locator("#temporal-question").fill("What changed between these two images?")
        with page.expect_response(lambda response: "/api/v1/temporal" in response.url and response.request.method == "POST") as awaited:
            page.locator("#run-temporal-change").click()
        response = awaited.value
        assert response.status == 200
        page.locator("#temporal-change-output").wait_for(state="visible")
        page.wait_for_function("!document.getElementById('run-temporal-change').disabled")
        output = page.locator("#temporal-change-output").inner_text()
        provenance = page.locator("#temporal-change-provenance").inner_text()
        assert "TEMPORAL_CHANGE_DESCRIPTION" in output and "Chg2Cap" in output and "change description:" in output
        assert "PRE_POST" in provenance and "generated_tokens" in provenance
        assert '"pair_id":"val_000001.png"' in requests[0]["post_data"]
        screenshot = OUT / "satquery-bitemporal-live-result.png"
        page.screenshot(path=str(screenshot), full_page=True)
        happy_console_errors, happy_page_errors, happy_failed = list(console_errors), list(page_errors), list(failed)
        happy_temporal_requests = len(requests)
        # Browser negative UI checks: errors are local client-side, not mocks.
        page.reload(wait_until="networkidle"); page.get_by_role("button", name="Bi-temporal").click()
        page.locator("#run-temporal-change").click(); missing_t1 = page.locator("#temporal-change-output").inner_text()
        page.locator("#temporal-t1").set_input_files(str(PAIR / "A" / "val_000001.png")); page.locator("#run-temporal-change").click(); missing_t2 = page.locator("#temporal-change-output").inner_text()
        page.locator("#temporal-t2").set_input_files(str(PAIR / "A" / "val_000001.png")); page.locator("#run-temporal-change").click()
        page.wait_for_timeout(400); identical = page.locator("#temporal-change-output").inner_text()
        unsupported = OUT / "unsupported.txt"; unsupported.write_text("not an image", encoding="utf-8")
        page.reload(wait_until="networkidle"); page.get_by_role("button", name="Bi-temporal").click()
        page.locator("#temporal-t1").set_input_files(str(unsupported)); page.locator("#temporal-t2").set_input_files(str(PAIR / "B" / "val_000001.png")); page.locator("#run-temporal-change").click()
        page.wait_for_timeout(400); unsupported_result = page.locator("#temporal-change-output").inner_text()
        temporal_requests_before_isolation = len(requests)
        page.get_by_role("button", name="Single image").click(); assert page.locator("#single-image-panel").is_visible()
        page.get_by_role("button", name="Optical + SAR", exact=True).click(); assert page.locator("#paired-v1-panel").is_visible()
        assert len(requests) == temporal_requests_before_isolation
        browser.close()
    (OUT / "playwright_observed.json").write_text(json.dumps({"browser": "Google Chrome", "headless": True, "url": BASE_URL,
        "output": output, "provenance": provenance, "requests": requests, "responses": responses,
        "happy_path": {"console_errors": happy_console_errors, "page_errors": happy_page_errors, "failed_requests": happy_failed,
                       "temporal_requests": happy_temporal_requests},
        "negative_case_browser_messages": {"console_errors": console_errors[len(happy_console_errors):], "page_errors": page_errors[len(happy_page_errors):], "failed_requests": failed[len(happy_failed):]},
        "negative": {"missing_t1": missing_t1, "missing_t2": missing_t2, "identical": identical, "unsupported": unsupported_result},
        "routing_isolation": {"single_image_temporal_requests": 0, "optical_sar_temporal_requests": 0, "bitemporal_happy_path_temporal_requests": happy_temporal_requests,
                              "negative_case_temporal_requests": temporal_requests_before_isolation - happy_temporal_requests}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": output, "response_status": response.status, "happy_path_console_errors": len(happy_console_errors),
                      "happy_path_page_errors": len(happy_page_errors), "happy_path_failed_requests": len(happy_failed), "screenshot": str(screenshot)}, indent=2))


if __name__ == "__main__":
    main()
