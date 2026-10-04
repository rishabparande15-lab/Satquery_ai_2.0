"""Local browser verification helper for the SIH demo controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright


BASE = "http://127.0.0.1:8012"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main() -> None:
    out = Path("artifacts/sih_demo_final")
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page()
        page.goto(BASE, wait_until="networkidle")
        page.locator("#view-home [data-view=analysis]").first.click()
        console, page_errors, failed = [], [], []
        page.on("console", lambda message: console.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("requestfailed", lambda request: failed.append(request.url))
        results = {}
        def run(name: str, selector: str):
            with page.expect_response(lambda response: response.url.endswith("/api/v1/demo/run") and response.request.method == "POST", timeout=900_000) as event:
                page.locator(selector).click()
            response = event.value
            body = response.json()
            page.wait_for_function(f"!document.querySelector('{selector}').disabled", timeout=900_000)
            if response.status != 200 or body.get("status") != "COMPLETED":
                raise RuntimeError(json.dumps(body))
            with page.expect_download(timeout=30_000) as download_event:
                page.locator({"s2": "#single-image-export", "scene": "#single-image-export", "sar": "#sar-vqa-export", "pair": "#v1-export", "temporal": "#temporal-change-export", "s2_revisit": "#single-image-export"}[name]).click()
            download_event.value.save_as(str(out / f"{name}.json"))
            results[name] = {"http": response.status, "route": body.get("route"), "status": body.get("status"), "answer": body.get("answer"), "warnings": body.get("warnings"), "confidence": body.get("confidence"), "evidence": bool(body.get("evidence")), "provenance": bool(body.get("provenance")), "trace": bool(body.get("execution_trace"))}

        if "--sar-only" in sys.argv:
            page.get_by_role("button", name="Use SAR Demo Sample").click()
            assert page.locator("#sar-vqa-question").input_value() == "Do parts of the image correspond to pastures?"
            run("sar", "#run-sar-vqa")
        else:
            page.get_by_role("button", name="Use S2 Demo Sample").click()
            assert page.locator("#single-image-question").input_value() == "Is water visible in this image?"
            run("s2", "#run-single-image")
            page.get_by_role("button", name="Use Scene Demo Sample").click()
            run("scene", "#run-single-image")
            page.get_by_role("button", name="Use SAR Demo Sample").click()
            assert page.locator("#sar-vqa-question").input_value() == "Do parts of the image correspond to pastures?"
            run("sar", "#run-sar-vqa")
            page.get_by_role("button", name="Use Optical + SAR Demo Pair").click()
            assert "Optical: Sentinel-2 loaded" in page.locator("#v1-input-mode").inner_text()
            run("pair", "#run-v1")
            page.get_by_role("button", name="Load Before + After Demo").click()
            assert "BEFORE (PRE) loaded" in page.locator("#temporal-input-mode").inner_text()
            run("temporal", "#run-temporal-change")
            page.get_by_role("button", name="Use S2 Demo Sample").click()
            assert page.locator("#single-image-question").input_value() == "Is water visible in this image?"
            run("s2_revisit", "#run-single-image")
        (out / "browser_audit.json").write_text(json.dumps({"results": results, "console_errors": console, "page_errors": page_errors, "failed_requests": failed}, indent=2), encoding="utf-8")
        print(json.dumps({"results": results, "console_errors": console, "page_errors": page_errors, "failed_requests": failed}, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()
