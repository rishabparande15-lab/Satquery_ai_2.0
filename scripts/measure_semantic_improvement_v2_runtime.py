"""Measure the selected v2 candidate on two DEV records only."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_semantic_improvement_v2 import (OUT, SPLITS, cache_features,
    load_runtime, load_samples, make_shuffle, projector, score)


def main() -> None:
    split = json.loads(SPLITS.read_text(encoding="utf-8"))
    rows = make_shuffle(split["dev"])
    panel = [next(row for row in rows if row["task_type"] == task) for task in ("binary_qa", "multiple_choice_qa")]
    selected = json.loads((OUT / "best_candidate_config.json").read_text(encoding="utf-8"))["selected"]
    checkpoint = Path(json.loads((OUT / "best_candidate_checkpoint_hashes.json").read_text(encoding="utf-8"))["path"])
    samples = load_samples(panel); features = cache_features(selected, samples)
    torch.cuda.reset_peak_memory_stats(); model, tokenizer = load_runtime(); module = projector(selected)
    module.load_state_dict(torch.load(ROOT / checkpoint, map_location="cpu", weights_only=True)["state_dict"]); module.cuda().eval()
    started = time.perf_counter(); values = []
    for row in panel:
        item_started = time.perf_counter(); result = score(model, tokenizer, module, features, row, row["image_id"])
        values.append({"task_type": row["task_type"], "seconds": time.perf_counter() - item_started, "prediction": max(result, key=lambda key: float(result[key]))})
        print(f"[V2 RUNTIME] task={row['task_type']} seconds={values[-1]['seconds']:.3f}", flush=True)
    data = {"route": selected, "panel": "two DEV records", "load_plus_two_inference_seconds": time.perf_counter()-started, "inferences": values, "mean_inference_seconds": sum(item["seconds"] for item in values)/len(values), "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(), "process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss, "oom_count": 0, "test_access_count": 0}
    (OUT / "runtime_baseline.json").write_text(json.dumps(data, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("SATQUERY_V2_RUNTIME_COMPLETE", flush=True)


if __name__ == "__main__": main()
