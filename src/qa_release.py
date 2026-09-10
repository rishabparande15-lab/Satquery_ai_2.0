"""Repeatable real-data release audit against the running loopback application."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import rasterio
import psutil
from .config import PROJECT_ROOT, get_settings
from .dataset_loader import discover_samples, load_sample, OPTICAL_BANDS, SAR_BANDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT_ROOT / "experiments" / "outputs" / "qa_release" / stamp
    root.mkdir(parents=True, exist_ok=False)
    audit = {"timestamp": stamp, "checks": [], "demo_runs": [], "browser": "Not verified: computer-use policy stopped browser automation because the current URL could not be determined."}
    def call(method, route, body=None, headers=None):
        if isinstance(body, dict):
            body = json.dumps(body)
            headers = {"Content-Type": "application/json", **(headers or {})}
        start = time.perf_counter()
        c = HTTPConnection("127.0.0.1", args.port, timeout=180)
        c.request(method, route, body, headers or {})
        r = c.getresponse()
        raw = r.read()
        status, hdrs = r.status, dict(r.getheaders())
        c.close()
        data = json.loads(raw) if raw and "application/json" in hdrs.get("Content-Type", "") else raw
        return status, data, hdrs, time.perf_counter() - start
    def check(label, condition):
        audit["checks"].append({"check": label, "passed": bool(condition)})
        if not condition:
            raise AssertionError(label)
    def memory():
        try:
            pid = next(c.pid for c in psutil.net_connections("tcp") if c.status == "LISTEN" and c.laddr.port == args.port)
            p = psutil.Process(pid)
            return {"rss_bytes": p.memory_info().rss, "cpu_seconds": sum(p.cpu_times()[:2])}
        except (psutil.Error, StopIteration):
            return {"status": "unavailable"}
    zip_path = get_settings().dataset_root.parent.parent / "bigearthnet-v2-three-samples.zip"
    def digest():
        with zip_path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    audit["zip_sha256_before"] = digest()
    for route in ("/", "/static/app.js", "/static/style.css", "/api/health", "/api/samples"):
        status, data, hdrs, runtime = call("GET", route)
        check("GET " + route, status == 200)
    audit["server_memory_before"] = memory()
    cases = [("optical", "Analyze this optical remote-sensing image and report the extracted features.", 52, 768),
             ("sar", "Analyze the SAR information in this region.", 9, 768),
             ("joint", "Perform a joint optical-SAR analysis and explain the extracted representation.", 62, 2304)]
    for repeat in range(2):
        for mode, query, physical, deep in cases:
            status, report, _, duration = call("POST", "/api/analyze", {"query": query, "sample_id": "61_39"})
            check(f"{mode} repeat {repeat} real inference", status == 200 and report["status"] == "completed")
            check(f"{mode} physical dimension", report["features"]["spectral"]["dimension"] == physical)
            check(f"{mode} CROMA dimension", report["features"]["deep"]["pooled_dimension"] == deep)
            check(f"{mode} modality", report["data_cube"]["modalities"] == (["optical", "sar"] if mode == "joint" else [mode]))
            check(f"{mode} trace", len(report["execution_trace"]) == 15)
            check(f"{mode} no prediction", report["model_results"]["prediction"] is None)
            if mode == "joint":
                check("actual joint fusion", report["features"]["hybrid"]["dimension"] == 192)
                check("joint encoding used", "joint_GAP" in report["features"]["deep"]["representations"])
            else:
                check("no spurious joint output", "joint_GAP" not in report["features"]["deep"]["representations"])
            check("finite strict JSON", bool(json.dumps(report, allow_nan=False)))
            check("no absolute internal paths", str(PROJECT_ROOT) not in json.dumps(report) and str(get_settings().dataset_root) not in json.dumps(report))
            download_status, downloaded, headers, _ = call("GET", "/api/report/" + report["analysis_id"])
            check("report download matches", download_status == 200 and downloaded == report and "attachment" in headers.get("Content-Disposition", ""))
            (root / f"{mode}-{repeat}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            audit["demo_runs"].append({"mode": mode, "repeat": repeat, "analysis_id": report["analysis_id"], "runtime_seconds": report["runtime_seconds"],
                                      "http_seconds": duration, "device": report["device"], "physical_dimension": physical, "deep_dimension": deep,
                                      "hybrid_dimension": report["features"]["hybrid"].get("dimension"), "model_reused": report["features"]["deep"]["model_reused"],
                                      "server_resources": memory()})
            print(json.dumps(audit["demo_runs"][-1]), flush=True)
    # Actual upload assembly with the same known local sample, in ephemeral QA files.
    item = load_sample(next(s for s in discover_samples(get_settings().dataset_root) if s.patch_id == "61_39"))
    with tempfile.TemporaryDirectory(prefix="satquery-qa-") as tmp:
        form = bytearray()
        boundary = "SatQueryReleaseBoundary"
        for mode, values, bands in (("optical", item.raw_optical, OPTICAL_BANDS), ("sar", item.raw_sar, SAR_BANDS)):
            path = Path(tmp) / (mode + ".tif")
            with rasterio.open(path, "w", driver="GTiff", count=len(bands), height=120, width=120, dtype="float32", crs=item.metadata["crs"], transform=rasterio.Affine(*item.metadata["transform"][:6])) as ds:
                ds.write(values)
                ds.descriptions = bands
            form.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="{mode}"; filename="{mode}.tif"\r\nContent-Type: image/tiff\r\n\r\n').encode() + path.read_bytes() + b"\r\n")
        form.extend(f"--{boundary}--\r\n".encode())
        status, uploaded, _, _ = call("POST", "/api/upload", bytes(form), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        check("actual GeoTIFF upload accepted", status == 200)
        status, report, _, _ = call("POST", "/api/analyze", {"query": "joint optical SAR", "files": uploaded["files"]})
        check("actual uploaded joint fusion", status == 200 and report["features"]["hybrid"]["dimension"] == 192)
        check("upload provenance", report["data_retrieval"]["source"] == "user-provided upload")
        (root / "uploaded-joint.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for name, request in (("temporal", {"query": "Compare this region between January 2020 and January 2025.", "sample_id": "61_39"}),
                          ("invalid-aoi", {"query": "optical", "sample_id": "61_39", "aoi": {"type": "point", "latitude": 100, "longitude": 0}}),
                          ("outside-aoi", {"query": "optical", "sample_id": "61_39", "aoi": {"type": "point", "latitude": 0, "longitude": 0}}),
                          ("invalid-date", {"query": "optical", "sample_id": "61_39", "start_date": "2025-02-30"}),
                          ("empty-input", {"query": "optical"})):
        status, report, _, _ = call("POST", "/api/analyze", request)
        check(name, status == 422 and not report["validation"]["valid"] and report["model_results"]["prediction"] is None)
        (root / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    status, error, _, _ = call("POST", "/api/analyze", {"query": ""})
    check("empty query error", status == 400 and error["error"]["code"] == "invalid_request")
    check("health after repeated runs", call("GET", "/api/health")[1]["busy"] is False)
    check("upload cleanup", not list((PROJECT_ROOT / "experiments" / "outputs" / "web_uploads").glob("request-*")))
    audit["zip_sha256_after"] = digest()
    check("original ZIP unchanged", audit["zip_sha256_before"] == audit["zip_sha256_after"])
    audit["status"] = "PASS WITH LIMITATIONS"
    audit["server_memory_after"] = memory()
    (root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("QA_REPORT=" + str(root / "audit.json"), flush=True)

if __name__ == "__main__":
    main()
