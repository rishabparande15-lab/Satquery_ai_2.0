"""Real Chrome verification of Phase 3X scene description."""
from __future__ import annotations
import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "sih" / "phase3x"
BASE = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8770")
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PATCH = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    requests, responses, console, page_errors, failed = [], [], [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed", lambda r: failed.append({"url": r.url, "failure": str(r.failure)}))
        page.on("request", lambda r: requests.append({"method": r.method, "url": r.url, "post_data": r.post_data}) if "/api/v1/query" in r.url else None)
        page.on("response", lambda r: responses.append({"status": r.status, "body": r.json()}) if "/api/v1/query" in r.url else None)
        page.goto(BASE, wait_until="networkidle")
        page.get_by_role("button", name="Single image").click()
        page.locator("#single-image-id").select_option(PATCH)
        page.locator("#single-image-task").select_option("SINGLE_IMAGE_SCENE_DESCRIPTION")
        page.locator("#single-image-question").fill("Describe this image.")
        with page.expect_response(lambda r: "/api/v1/query" in r.url and r.request.method == "POST") as event:
            page.locator("#run-single-image").click()
        response = event.value; body = response.json()
        assert response.status == 200 and body["status"] == "COMPLETED"
        assert body["route"] == "SINGLE_IMAGE_SCENE_DESCRIPTION" and body["answer"].strip()
        page.wait_for_function("!document.getElementById('run-single-image').disabled")
        output = page.locator("#single-image-output").inner_text()
        assert "SINGLE_IMAGE_SCENE_DESCRIPTION" in output and "description:" in output
        assert page.locator("#single-image-provenance-details").get_attribute("open") is not None
        with page.expect_download() as download_event: page.locator("#single-image-export").click()
        download = download_event.value.suggested_filename
        page.screenshot(path=str(OUT / "satquery-phase3x-scene-description-live.png"), full_page=True)
        browser.close()
    result = {"browser": "Google Chrome", "headless": True, "url": BASE, "requests": requests, "responses": responses,
              "output": output, "download": download, "console_errors": console, "page_errors": page_errors, "failed_requests": failed,
              "test_access": 0}
    (OUT / "phase3x_scene_browser_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"http_status": response.status, "route": body["route"], "description": body["answer"], "console_errors": len(console), "page_errors": len(page_errors)}, indent=2))

if __name__ == "__main__": main()
