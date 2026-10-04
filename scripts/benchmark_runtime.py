"""Run the fixed, non-test SatQuery runtime benchmark with live progress."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import psutil
import torch

from src.api import run_demo_query
from src.specialist_runtime import RUNTIME

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts" / "runtime_optimization"
PATH_PATTERN = re.compile(r"(?i)(?:[a-z]:[\\\\/]|/(?:home|tmp)/|models--|snapshots/)")


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()
                if str(key) not in {"runtime_seconds", "request_id", "analysis_id", "timestamp", "duration_ms"}}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and PATH_PATTERN.search(value):
        return "<SANITIZED_RUNTIME_PATH>"
    return value


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(_sanitize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gpu() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _run_route(spec: dict[str, Any], kind: str, index: int, total: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = _gpu()
    ram_before = psutil.Process(os.getpid()).memory_info().rss
    started = time.perf_counter()
    result = run_demo_query(spec["fixture_id"], spec["question"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after = _gpu()
    ram_after = psutil.Process(os.getpid()).memory_info().rss
    route = str(result.get("route") or result.get("task") or "")
    message = (
        f"[BENCH {index}/{total}] route={route} state={kind} total_ms={elapsed * 1000:.1f} "
        f"gpu_alloc_mb={after['peak_allocated'] / 1048576:.1f} "
        f"gpu_reserved_mb={after['peak_reserved'] / 1048576:.1f} ram_mb={ram_after / 1048576:.1f}"
    )
    print(message, flush=True)
    with (OUTPUT_ROOT / "run.log").open("a", encoding="utf-8") as log:
        log.write(message + "\n")
    output = _sanitize(result)
    metric = {
        "fixture_id": spec["fixture_id"],
        "expected_route": spec["route"],
        "actual_route": route,
        "state": kind,
        "latency_ms": round(elapsed * 1000, 3),
        "gpu_before": before,
        "gpu_after": after,
        "ram_before_bytes": ram_before,
        "ram_after_bytes": ram_after,
        "runtime_status": RUNTIME.status(),
        "status": result.get("status"),
        "output_fingerprint": _fingerprint(result),
    }
    return metric, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cold", type=int, default=1)
    parser.add_argument("--warm", type=int, default=3)
    args = parser.parse_args()
    config = json.loads((OUTPUT_ROOT / "benchmark_config.json").read_text(encoding="utf-8"))
    routes = config["routes"]
    metrics: dict[str, Any] = {"label": args.label, "routes": {}, "started_at_unix": time.time()}
    outputs: dict[str, Any] = {"label": args.label, "routes": {}}
    trace: list[dict[str, Any]] = []
    total = len(routes) * (args.cold + args.warm)
    step = 0
    for spec in routes:
        RUNTIME.release_all()
        route_metrics: list[dict[str, Any]] = []
        route_outputs: list[dict[str, Any]] = []
        for run_index in range(args.cold + args.warm):
            step += 1
            kind = "cold" if run_index < args.cold else "warm"
            metric, output = _run_route(spec, kind, step, total)
            route_metrics.append(metric)
            route_outputs.append(output)
            trace.append({"fixture_id": spec["fixture_id"], "state": kind, "gpu": metric["gpu_after"],
                          "ram_after_bytes": metric["ram_after_bytes"], "resident": metric["runtime_status"]["resident_specialist"]})
        metrics["routes"][spec["route"]] = route_metrics
        outputs["routes"][spec["route"]] = route_outputs
    RUNTIME.release_all()
    metrics["finished_at_unix"] = time.time()
    metrics["final_runtime_status"] = RUNTIME.status()
    if args.label == "baseline":
        destination = OUTPUT_ROOT
        stem = "baseline"
    elif args.label.startswith("candidate_"):
        destination = OUTPUT_ROOT / "candidate_results" / args.label
        stem = args.label
    else:
        destination = OUTPUT_ROOT
        stem = args.label
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (destination / f"{stem}_outputs.json").write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    (destination / f"{stem}_memory_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
