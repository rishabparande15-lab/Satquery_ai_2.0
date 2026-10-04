"""Verify the fixed demo selectors through the loopback HTTP API."""
from __future__ import annotations
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
out = {"health": None, "ready": None, "routes": {}}
for name, url in (("health", "http://127.0.0.1:8011/api/v1/health"), ("ready", "http://127.0.0.1:8011/api/v1/ready")):
    with urlopen(url, timeout=30) as response:
        out[name] = {"http": response.status, "body": json.loads(response.read())}
for demo_id in ("s2_vqa", "sar_vqa", "optical_sar", "scene", "temporal"):
    request = Request("http://127.0.0.1:8011/api/v1/demo/run",
        data=json.dumps({"demo_id": demo_id}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        body = json.loads(response.read())
        out["routes"][demo_id] = {"http": response.status, "latency_ms": round((time.perf_counter()-started)*1000,3),
            "route": body.get("route"), "status": body.get("status"), "answer": body.get("answer") or body.get("generated_response") or body.get("change_description"),
            "warnings": body.get("warnings", [])}
    print(f"[API] demo={demo_id} http={out['routes'][demo_id]['http']} route={out['routes'][demo_id]['route']} latency_ms={out['routes'][demo_id]['latency_ms']}", flush=True)
(ROOT/"artifacts/runtime_optimization/api_demo_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
