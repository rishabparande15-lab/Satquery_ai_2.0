"""Measure the frozen constrained-decoding candidate on a small validation panel.

This is deliberately a separate, post-comparison measurement so it cannot
overwrite the completed scientific records.  It reuses only predeclared
validation identities and executes one binary and one MCQ record per route.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_semantic_improvement import (OUT, ROUTES, close, constrained,
    dump, load_panel, make_controller, validation_samples)


def main() -> None:
    rows = load_panel()
    panel = [next(row for row in rows if row["task_type"] == task) for task in ("binary_qa", "multiple_choice_qa")]
    samples = validation_samples(panel)
    metrics = {"panel_records_per_route": 2, "test_access_count": 0, "routes": {}}
    for route in ROUTES:
        torch.cuda.reset_peak_memory_stats()
        controller = make_controller(route)
        started = time.perf_counter()
        controller.load()
        latencies = []
        for row in panel:
            one_started = time.perf_counter()
            result = constrained(controller, route, row, samples[row["image_id"]])
            latencies.append({"task_type": row["task_type"], "seconds": time.perf_counter() - one_started, "prediction": result["prediction"]})
            print(f"[RUNTIME] route={route} task={row['task_type']} seconds={latencies[-1]['seconds']:.3f}", flush=True)
        metrics["routes"][route] = {
            "load_plus_two_inferences_seconds": time.perf_counter() - started,
            "inference_seconds": latencies,
            "mean_inference_seconds": sum(item["seconds"] for item in latencies) / len(latencies),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "oom_count": 0,
        }
        close(controller)
    dump("runtime_metrics.json", metrics)
    print("SATQUERY_SEMANTIC_RUNTIME_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
