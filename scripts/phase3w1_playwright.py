"""Real unified-browser completion using the two recovered approved inputs."""
from __future__ import annotations
import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "agent" / "phase3w1_browser_completion"
BASE = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8768")
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
SINGLE = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
PAIR = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"

def body(response):
    return response.json()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    requests, responses, console, page_errors, failed = [], [], [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed", lambda r: failed.append({"url": r.url, "failure": str(r.failure)}))
        page.on("request", lambda r: requests.append({"method": r.method, "url": r.url, "post_data": r.post_data}) if "/api/v1/query" in r.url else None)
        page.on("response", lambda r: responses.append({"status": r.status, "body": body(r)}) if "/api/v1/query" in r.url else None)
        page.goto(BASE, wait_until="networkidle")

        # Real single-image VQA through the unified UI.
        page.get_by_role("button", name="Single image").click()
        page.locator("#single-image-id").select_option(SINGLE)
        page.locator("#single-image-task").select_option("SINGLE_IMAGE_VQA")
        page.locator("#single-image-question").fill("Is water visible in this image?")
        with page.expect_response(lambda r: "/api/v1/query" in r.url and r.request.method == "POST") as event:
            page.locator("#run-single-image").click()
        single_response = event.value
        assert single_response.status == 200 and single_response.json()["route"] == "SINGLE_IMAGE_VQA"
        page.wait_for_function("!document.getElementById('run-single-image').disabled")
        single_output = page.locator("#single-image-output").inner_text()
        assert "SINGLE_IMAGE_VQA" in single_output and "answer:" in single_output
        assert page.locator("#single-image-provenance-details").get_attribute("open") is not None
        with page.expect_download() as download_event: page.locator("#single-image-export").click()
        single_download = download_event.value.suggested_filename
        page.screenshot(path=str(OUT / "satquery-phase3w1-single-image-live.png"), full_page=True)

        # Same real input; selected grounding must remain blocked and must not call a VQA specialist.
        page.locator("#single-image-task").select_option("SINGLE_IMAGE_GROUNDING")
        page.locator("#single-image-question").fill("Highlight the water body.")
        with page.expect_response(lambda r: "/api/v1/query" in r.url and r.request.method == "POST") as event:
            page.locator("#run-single-image").click()
        grounding_response = event.value
        assert grounding_response.status == 200 and grounding_response.json()["route"] == "SINGLE_IMAGE_GROUNDING"
        assert grounding_response.json()["status"] == "BLOCKED"
        page.wait_for_function("!document.getElementById('run-single-image').disabled")
        grounding_output = page.locator("#single-image-output").inner_text()
        assert "Grounding unavailable" in grounding_output
        page.screenshot(path=str(OUT / "satquery-phase3w1-grounding-blocked.png"), full_page=True)

        # Existing panel options are intentionally bounded; insert only the already-verified local pair for this test fixture.
        page.get_by_role("button", name="Optical + SAR", exact=True).click()
        page.locator("#v1-patch-id").evaluate("(select, value) => { if (![...select.options].some(o => o.value === value)) { const o = new Option(value, value, true, true); select.add(o); } select.value = value; }", PAIR)
        page.locator("#v1-task-type").select_option("binary_qa")
        page.locator("#v1-question").fill("Use the SAR and optical information together. Is water visible?")
        with page.expect_response(lambda r: "/api/v1/query" in r.url and r.request.method == "POST") as event:
            page.locator("#run-v1").click()
        paired_response = event.value
        assert paired_response.status == 200 and paired_response.json()["route"] == "OPTICAL_SAR_ANALYSIS"
        page.wait_for_function("!document.getElementById('run-v1').disabled")
        paired_output = page.locator("#v1-output").inner_text()
        assert "OPTICAL_SAR_ANALYSIS" in paired_output and "answer:" in paired_output
        with page.expect_download() as download_event: page.locator("#v1-export").click()
        paired_download = download_event.value.suggested_filename
        page.screenshot(path=str(OUT / "satquery-phase3w1-optical-sar-live.png"), full_page=True)
        browser.close()
    result = {"browser": "Google Chrome", "headless": True, "url": BASE, "requests": requests, "responses": responses,
              "single_image": {"output": single_output, "download": single_download},
              "grounding": {"output": grounding_output}, "optical_sar": {"output": paired_output, "download": paired_download},
              "console_errors": console, "page_errors": page_errors, "failed_requests": failed}
    (OUT / "phase3w1_browser_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"single": single_response.status, "grounding": grounding_response.status, "paired": paired_response.status, "console": len(console), "page": len(page_errors)}, indent=2))

if __name__ == "__main__": main()
