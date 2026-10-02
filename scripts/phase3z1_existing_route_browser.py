"""Recheck the existing real browser routes without changing their models."""
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
import rasterio
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.dataset_loader import OPTICAL_BANDS, discover_s2_samples
OUT = ROOT / "artifacts" / "final" / "input" / "phase3z1"
BASE = os.environ["SATQUERY_BASE_URL"]
MODE = os.environ["SATQUERY_ROUTE_MODE"]
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
DATASET = Path(os.environ["DATASET_ROOT"])
LEVIR = ROOT / "datasets" / "levir_cc" / "development_smoke" / "images" / "val"

def stack_first_train_s2(destination: Path) -> str:
    sample = next(item for item in discover_s2_samples(DATASET, allowed_splits=("train",)) if len(item.optical_paths) == 12)
    with rasterio.open(sample.optical_paths["B02"]) as reference:
        profile = reference.profile.copy(); profile.update(count=12)
        with rasterio.open(destination, "w", **profile) as out:
            for index, band in enumerate(OPTICAL_BANDS, 1):
                with rasterio.open(sample.optical_paths[band]) as source:
                    out.write(source.read(1), index); out.set_band_description(index, band)
    return sample.patch_id

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    console, page_errors, failed = [], [], []
    with tempfile.TemporaryDirectory(prefix="phase3z1-existing-") as temp:
        temp_path = Path(temp)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.on("console", lambda message: console.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on("requestfailed", lambda request: failed.append({"url": request.url, "failure": str(request.failure)}))
            page.goto(BASE, wait_until="networkidle")
            if MODE == "external_s2":
                fixture = temp_path / "approved-train-s2.tif"; source_id = stack_first_train_s2(fixture)
                page.get_by_role("button", name="Single image").click()
                page.locator("#single-image-upload").set_input_files(str(fixture))
                page.locator("#single-image-sensor-declared").check(); page.locator("#single-image-band-order").check()
                page.locator("#single-image-task").select_option("SINGLE_IMAGE_VQA")
                page.locator("#single-image-question").fill("Is water visible in this image?")
                button, output_id, expected = "#run-single-image", "#single-image-output", "SINGLE_IMAGE_VQA"
            elif MODE == "optical_sar":
                pair = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"; source_id = pair
                page.get_by_role("button", name="Optical + SAR", exact=True).click()
                page.locator("#v1-patch-id").evaluate("(select,value)=>{if(![...select.options].some(option=>option.value===value)) select.add(new Option(value,value)); select.value=value}", pair)
                page.locator("#v1-question").fill("Use the SAR and optical information together. Is water visible?")
                button, output_id, expected = "#run-v1", "#v1-output", "OPTICAL_SAR_ANALYSIS"
            else:
                source_id = "LEVIR_CC validation val_000001.png"
                page.get_by_role("button", name="Bi-temporal").click()
                page.locator("#temporal-t1").set_input_files(str(LEVIR / "A" / "val_000001.png")); page.locator("#temporal-t2").set_input_files(str(LEVIR / "B" / "val_000001.png"))
                page.locator("#temporal-question").fill("What changed between these two images?")
                button, output_id, expected = "#run-temporal-change", "#temporal-change-output", "TEMPORAL_CHANGE_DESCRIPTION"
            with page.expect_response(lambda response: "/api/v1/query" in response.url and response.request.method == "POST", timeout=180000) as event:
                page.locator(button).click()
            response = event.value; body = response.json(); page.wait_for_timeout(150)
            assert response.status == 200 and body["status"] == "COMPLETED" and body["route"] == expected, body
            assert not page.locator(button).is_disabled(); output = page.locator(output_id).inner_text(); assert expected in output
            browser.close()
    result = {"mode": MODE, "source": source_id, "route": body["route"], "status": body["status"], "answer": body.get("answer"), "http_status": response.status, "ui_output": output, "console_errors": console, "page_errors": page_errors, "failed_requests": failed, "test_access": 0}
    (OUT / f"phase3z1_browser_{MODE}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
