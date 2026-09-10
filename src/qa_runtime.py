"""Startup and stored temporal-export checks; no retrieval or training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import rasterio
import numpy as np
from .config import PROJECT_ROOT
from .input_validation import check_grid


def main():
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    began = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-m", "src.api", "--host", "127.0.0.1", "--port", str(port)],
                               cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    report = {}
    try:
        while time.perf_counter() - began < 30:
            if process.poll() is not None:
                raise RuntimeError("Startup process exited before health check")
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
                    if response.status == 200:
                        report["startup_seconds"] = time.perf_counter() - began
                        break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("Startup exceeded 30 seconds")
    finally:
        process.terminate()
        stdout, stderr = process.communicate(timeout=15)
        report["startup_traceback"] = b"Traceback" in stdout + stderr
    native = PROJECT_ROOT / "experiments" / "outputs" / "real_gee" / "gee_temporal"
    report["native_temporal_exports"] = []
    for sensor, bands in (("sentinel_2", ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]),
                          ("sentinel_1", ["VV","VH"])):
        for band in bands:
            meta, paths = [], []
            for period in ("before", "after"):
                path = native / f"{period}_{sensor}" / f"{period}_{sensor}.{band}.tif"
                paths.append(path)
                with rasterio.open(path) as ds:
                    meta.append({"crs": str(ds.crs), "resolution": list(ds.res), "bounds": list(ds.bounds), "transform": list(ds.transform), "shape": [ds.height,ds.width]})
            correspondence = check_grid(*meta)
            finite = None
            if all(correspondence.values()):
                finite = []
                for path in paths:
                    with rasterio.open(path) as ds:
                        finite.append(bool(np.isfinite(ds.read(out_dtype="float32", masked=True).filled(np.nan)).all()))
            report["native_temporal_exports"].append({"sensor":sensor,"band":band,"grid_checks":correspondence,"finite_before_after":finite,"before_metadata":meta[0],"after_metadata":meta[1]})
    stored = json.loads((native / "execution_report.json").read_text(encoding="utf-8"))
    report["source"] = "Previously stored local GEE exports; no new provider retrieval performed"
    report["stored_acquisition_dates"] = {sensor: [item["acquisition_date"] for item in stored[sensor]] for sensor in ("sentinel_1_images","sentinel_2_images")}
    report["temporal_analysis_status"] = "Grid and finite-pixel audit only. No mask, area change, accuracy or change-detection claim."
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = PROJECT_ROOT / "experiments" / "outputs" / "qa_release" / f"runtime-{stamp}.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"startup_seconds":report["startup_seconds"],"startup_traceback":report["startup_traceback"],
                     "temporal_grids_matched":sum(all(item["grid_checks"].values()) for item in report["native_temporal_exports"]),
                     "temporal_band_pairs":len(report["native_temporal_exports"]),"report":str(destination)}))

if __name__ == "__main__":
    main()
