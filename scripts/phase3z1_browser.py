"""Real browser verification for Phase 3Z.1 external-input integration.

Uses one admitted LEVIR-CC validation RGB image and synthetic bounded TIFFs
only to exercise the inspection-only, model-blocked sensor contracts.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from playwright.sync_api import sync_playwright
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "input" / "phase3z1"
LEVIR_RGB = ROOT / "datasets" / "levir_cc" / "development_smoke" / "images" / "val" / "A" / "val_000001.png"
BASE = os.environ["SATQUERY_BASE_URL"]
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def make_tiff(path: Path, bands: int) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=bands, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(0, 100, 10, 10)) as dst:
        dst.write(np.ones((bands, 10, 10), dtype=np.float32))


def inspect(page, file: Path, *, sensor: str, modality: str, role: str, expected: str) -> str:
    page.locator("#external-raster-upload").set_input_files(str(file))
    page.locator("#external-sensor").select_option(sensor)
    page.locator("#external-modality").select_option(modality)
    page.locator("#external-role").select_option(role)
    page.locator("#external-route").select_option("")
    page.locator("#inspect-external-raster").click()
    page.locator("#inspect-external-raster").wait_for(state="attached")
    page.wait_for_timeout(150)
    assert not page.locator("#inspect-external-raster").is_disabled()
    text = page.locator("#external-raster-output").inner_text()
    assert "FILE INSPECTION: SUCCESS" in text and expected in text, text
    return text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert LEVIR_RGB.is_file(), "Only the admitted validation RGB fixture may be used."
    console, page_errors, failed, responses = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="phase3z1-browser-") as temp_dir:
        temp = Path(temp_dir)
        cart, risat, ambiguous = temp / "cartosat.tif", temp / "risat.tif", temp / "ambiguous-8band.tif"
        make_tiff(cart, 3); make_tiff(risat, 2); make_tiff(ambiguous, 8)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1500})
            page.on("console", lambda message: console.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on("requestfailed", lambda request: failed.append({"url": request.url, "failure": str(request.failure)}))
            page.on("response", lambda response: responses.append({"url": response.url, "status": response.status}) if "/api/" in response.url else None)
            page.goto(BASE, wait_until="networkidle")

            page.get_by_role("button", name="Single image").click()
            page.locator("#single-image-task").select_option("SINGLE_IMAGE_SCENE_DESCRIPTION")
            page.locator("#scene-image-upload").set_input_files(str(LEVIR_RGB))
            page.locator("#scene-upload-sensor").select_option("generic-rgb")
            page.locator("#single-image-question").fill("Describe this image.")
            with page.expect_response(lambda r: "/api/v1/query" in r.url and r.request.method == "POST", timeout=180000) as event:
                page.locator("#run-single-image").click()
            scene_response = event.value
            scene = scene_response.json()
            page.wait_for_timeout(150)
            assert not page.locator("#run-single-image").is_disabled()
            scene_text = page.locator("#single-image-output").inner_text()
            assert scene_response.status == 200 and scene["status"] == "COMPLETED" and scene["route"] == "SINGLE_IMAGE_SCENE_DESCRIPTION", scene
            assert isinstance(scene.get("answer"), str) and scene["answer"].strip() and "description:" in scene_text

            cart_text = inspect(page, cart, sensor="cartosat-2s", modality="optical", role="SINGLE", expected="SENSOR_DOMAIN_NOT_VALIDATED")
            risat_text = inspect(page, risat, sensor="risat", modality="sar", role="SINGLE", expected="SENSOR_DOMAIN_NOT_VALIDATED")
            ambiguous_text = inspect(page, ambiguous, sensor="", modality="multispectral", role="SINGLE", expected="MISSING_SENSOR_DECLARATION")
            page.screenshot(path=str(OUT / "satquery-generic-input-live-result.png"), full_page=True)
            browser.close()
    result = {
        "browser": "Google Chrome", "url": BASE, "test_access": 0,
        "generic_rgb": {"source": "LEVIR_CC validation A/val_000001.png", "http_status": scene_response.status, "route": scene["route"], "status": scene["status"], "answer": scene["answer"], "ui_output": scene_text},
        "cartosat": {"ui_output": cart_text, "code": "SENSOR_DOMAIN_NOT_VALIDATED"},
        "risat": {"ui_output": risat_text, "code": "SENSOR_DOMAIN_NOT_VALIDATED"},
        "ambiguous_multiband": {"ui_output": ambiguous_text, "code": "MISSING_SENSOR_DECLARATION"},
        "console_errors": console, "page_errors": page_errors, "failed_requests": failed, "responses": responses,
    }
    (OUT / "phase3z1_browser_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generic_rgb": result["generic_rgb"], "cartosat": result["cartosat"]["code"], "risat": result["risat"]["code"], "ambiguous": result["ambiguous_multiband"]["code"]}, indent=2))


if __name__ == "__main__":
    main()
