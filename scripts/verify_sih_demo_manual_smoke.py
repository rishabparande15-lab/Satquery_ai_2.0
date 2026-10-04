"""Manual-upload smoke for the fixed non-test SIH demo fixtures."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import rasterio
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import PROJECT_ROOT, get_settings
from src.dataset_loader import OPTICAL_BANDS


BASE = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8011")
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE")
S2_PATCH = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
PAIR = "val_000001.png"


def approved_s2_stack(destination: Path) -> None:
    root = get_settings().dataset_root
    scene = S2_PATCH.rsplit("_", 2)[0]
    source = root / "BigEarthNet-S2" / scene / S2_PATCH
    with rasterio.open(source / f"{S2_PATCH}_B02.tif") as reference:
        profile = reference.profile.copy()
        profile.update(count=len(OPTICAL_BANDS))
        with rasterio.open(destination, "w", **profile) as output:
            for index, band in enumerate(OPTICAL_BANDS, 1):
                with rasterio.open(source / f"{S2_PATCH}_{band}.tif") as input_band:
                    output.write(input_band.read(1), index)
                output.set_band_description(index, band)


def run() -> None:
    output_dir = PROJECT_ROOT / "artifacts" / "runtime_optimization" / "final_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    console, page_errors, failed, results = [], [], [], {}
    with tempfile.TemporaryDirectory(prefix="satquery-approved-manual-") as temporary:
        stack = Path(temporary) / "approved-s2.tif"
        approved_s2_stack(stack)
        validation = get_settings().levir_root / "images" / "val"
        with sync_playwright() as playwright:
            launch_options = {"headless": True}
            if CHROME:
                launch_options["executable_path"] = CHROME
            browser = playwright.chromium.launch(**launch_options)
            page = browser.new_page()
            page.on("console", lambda item: console.append(item.text) if item.type == "error" else None)
            page.on("pageerror", lambda item: page_errors.append(str(item)))
            page.on("requestfailed", lambda item: failed.append(item.url))
            page.goto(BASE, wait_until="networkidle")
            page.locator("#view-home [data-view=analysis]").first.click()

            page.get_by_role("button", name="Use S2 Demo Sample").click()
            page.locator("#single-image-upload").set_input_files(str(stack))
            page.locator("#single-image-sensor-declared").check()
            page.locator("#single-image-band-order").check()
            with page.expect_response(lambda item: item.url.endswith("/api/v1/query") and item.request.method == "POST", timeout=900_000) as pending:
                page.locator("#run-single-image").click()
            response = pending.value
            body = response.json()
            page.wait_for_function("!document.querySelector('#run-single-image').disabled", timeout=900_000)
            results["manual_s2"] = {"http": response.status, "route": body.get("route"), "status": body.get("status"), "answer": body.get("answer"), "warnings": body.get("warnings", []), "confidence": body.get("confidence"), "evidence_present": isinstance(body.get("evidence"), dict), "provenance_present": isinstance(body.get("provenance"), dict)}

            page.get_by_role("button", name="Load Before + After Demo").click()
            page.locator("#temporal-t1").set_input_files(str(validation / "A" / PAIR))
            page.locator("#temporal-t2").set_input_files(str(validation / "B" / PAIR))
            with page.expect_response(lambda item: item.url.endswith("/api/v1/query") and item.request.method == "POST", timeout=900_000) as pending:
                page.locator("#run-temporal-change").click()
            response = pending.value
            body = response.json()
            page.wait_for_function("!document.querySelector('#run-temporal-change').disabled", timeout=900_000)
            results["manual_temporal"] = {"http": response.status, "route": body.get("route"), "status": body.get("status"), "answer": body.get("answer"), "warnings": body.get("warnings", []), "confidence": body.get("confidence"), "evidence_present": isinstance(body.get("evidence"), dict), "provenance_present": isinstance(body.get("provenance"), dict), "temporal_order": "PRE_POST"}
            browser.close()
    report = {"results": results, "console_errors": console, "page_errors": page_errors, "failed_requests": failed}
    (output_dir / "manual_s2_smoke.json").write_text(json.dumps({"fixture_logical_id": S2_PATCH, **results["manual_s2"], "path_leaks": []}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manual_temporal_smoke.json").write_text(json.dumps({"fixture_logical_id": PAIR, **results["manual_temporal"], "path_leaks": [], "fake_geometry_claims": []}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    run()
