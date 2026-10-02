"""One real unified-browser SIH rehearsal flow selected by SATQUERY_DEMO."""
from __future__ import annotations
import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "sih" / "phase3x1_demo_rehearsal"
BASE = os.environ["SATQUERY_BASE_URL"]
DEMO = os.environ["SATQUERY_DEMO"]
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
SINGLE = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
PAIR = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
LEVIR = ROOT / "datasets" / "levir_cc" / "development_smoke" / "images" / "val"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    console, page_errors, failed, requests, responses = [], [], [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed", lambda r: failed.append({"url": r.url, "failure": str(r.failure)}))
        page.on("request", lambda r: requests.append({"method":r.method,"url":r.url,"post_data":r.post_data}) if "/api/v1/query" in r.url else None)
        page.on("response", lambda r: responses.append({"status":r.status,"body":r.json()}) if "/api/v1/query" in r.url else None)
        page.goto(BASE, wait_until="networkidle")
        if DEMO in {"vqa", "scene", "grounding"}:
            page.get_by_role("button", name="Single image").click(); page.locator("#single-image-id").select_option(SINGLE)
            if DEMO == "vqa":
                page.locator("#single-image-task").select_option("SINGLE_IMAGE_VQA"); page.locator("#single-image-question").fill("Is water visible in this image?")
                expected, status, shot = "SINGLE_IMAGE_VQA", "COMPLETED", "satquery-final-demo-1-vqa.png"
            elif DEMO == "scene":
                page.locator("#single-image-task").select_option("SINGLE_IMAGE_SCENE_DESCRIPTION"); page.locator("#single-image-question").fill("Describe this image.")
                expected, status, shot = "SINGLE_IMAGE_SCENE_DESCRIPTION", "COMPLETED", "satquery-final-demo-4-scene-description.png"
            else:
                page.locator("#single-image-task").select_option("SINGLE_IMAGE_GROUNDING"); page.locator("#single-image-question").fill("Highlight the water body.")
                expected, status, shot = "SINGLE_IMAGE_GROUNDING", "BLOCKED", "satquery-final-demo-grounding-blocked.png"
            button, output_id, export_id = "#run-single-image", "#single-image-output", "#single-image-export"
        elif DEMO == "optical_sar":
            page.get_by_role("button", name="Optical + SAR", exact=True).click()
            page.locator("#v1-patch-id").evaluate("(s,v)=>{if(![...s.options].some(o=>o.value===v)){s.add(new Option(v,v))}s.value=v}", PAIR)
            page.locator("#v1-task-type").select_option("binary_qa"); page.locator("#v1-question").fill("Use the SAR and optical information together. Is water visible?")
            expected, status, shot = "OPTICAL_SAR_ANALYSIS", "COMPLETED", "satquery-final-demo-2-optical-sar.png"; button, output_id, export_id = "#run-v1", "#v1-output", "#v1-export"
        else:
            page.get_by_role("button", name="Bi-temporal").click()
            page.locator("#temporal-t1").set_input_files(str(LEVIR / "A" / "val_000001.png")); page.locator("#temporal-t2").set_input_files(str(LEVIR / "B" / "val_000001.png"))
            page.locator("#temporal-question").fill("What changed between these two images?")
            expected, status, shot = "TEMPORAL_CHANGE_DESCRIPTION", "COMPLETED", "satquery-final-demo-3-temporal.png"; button, output_id, export_id = "#run-temporal-change", "#temporal-change-output", "#temporal-change-export"
        with page.expect_response(lambda r: "/api/v1/query" in r.url and r.request.method == "POST", timeout=120000) as event: page.locator(button).click()
        response = event.value; body = response.json(); assert response.status == 200 and body["route"] == expected and body["status"] == status
        page.wait_for_function(f"!document.querySelector('{button}').disabled")
        output = page.locator(output_id).inner_text(); assert expected in output
        download = None
        if status == "COMPLETED":
            with page.expect_download() as dl: page.locator(export_id).click()
            download = dl.value.suggested_filename
        page.screenshot(path=str(OUT / shot), full_page=True); browser.close()
    result={"demo":DEMO,"browser":"Google Chrome","url":BASE,"http_status":response.status,"route":body["route"],"status":body["status"],"answer":body.get("answer"),"output":output,"download":download,"requests":requests,"responses":responses,"console_errors":console,"page_errors":page_errors,"failed_requests":failed,"test_access":0}
    (OUT / f"phase3x1_demo_{DEMO}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("demo","route","status","answer","download")},indent=2))
if __name__ == "__main__": main()
