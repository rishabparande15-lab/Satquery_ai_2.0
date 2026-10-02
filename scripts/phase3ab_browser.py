"""Real-browser inspection-only evidence verification; no model route is invoked."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "evidence" / "phase3ab"
BASE = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8794")
CHROME = os.environ.get("SATQUERY_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")

def make(path, count):
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=count, dtype="float32", crs="EPSG:4326", transform=from_origin(77, 29, .01, .01)) as dst: dst.write(np.ones((count, 8, 8), dtype="float32"))

def main():
    errors=[]; page_errors=[]
    with tempfile.TemporaryDirectory() as temp:
        cart=Path(temp)/"cart.tif"; risat=Path(temp)/"risat.tif"; make(cart,3); make(risat,2)
        with sync_playwright() as p:
            b=p.chromium.launch(executable_path=CHROME, headless=True); page=b.new_page(); page.on("console", lambda m: errors.append(m.text) if m.type=="error" else None); page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(BASE, wait_until="networkidle")
            observed={}
            for path, sensor, modality, key in [(cart,"cartosat-2s","optical","cartosat"),(risat,"risat","sar","risat")]:
                page.locator("#external-raster-upload").set_input_files(str(path)); page.locator("#external-sensor").select_option(sensor); page.locator("#external-modality").select_option(modality); page.locator("#inspect-external-raster").click(); page.wait_for_timeout(300)
                text=page.locator("#external-raster-evidence").inner_text(); assert "IMAGE STATISTICS" in text and "EPSG:4326" in text, text
                observed[key]=text
            page.screenshot(path=str(OUT/"satquery-geospatial-evidence-live.png"), full_page=True); b.close()
    (OUT/"phase3ab_browser_results.json").write_text(json.dumps({"status":"PASS","browser":"Google Chrome","url":BASE,"cartosat":observed["cartosat"],"risat":observed["risat"],"console_errors":errors,"page_errors":page_errors,"test_access":0},indent=2),encoding="utf-8")
if __name__=="__main__": main()
