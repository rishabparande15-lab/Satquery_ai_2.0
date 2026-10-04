"""Development-only semantic comparison for the frozen SatQuery VQA routes.

Uses the existing validation-only paired manifest, filters metadata to the
predeclared identities, and never enumerates or reads held-out test records.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import PROJECT_ROOT, get_settings
from src.dataset_loader import OPTICAL_BANDS, SAR_BANDS, Sample, load_sample
from src.eo_vlm.image_conditioned_objective import target_sequence_log_probability
from src.eo_vlm.training import build_qwen_visual_token_batch
from src.preprocessing import _robust_channel_scale
from src.satquery_v1 import SatQueryV1Controller
from src.single_image_sar_vqa import SingleImageSARVQAController
from src.single_image_vqa import SingleImageVQAController


OUT = PROJECT_ROOT / "artifacts" / "semantic_improvement"
MANIFEST = PROJECT_ROOT / "artifacts" / "training" / "phase3q" / "phase3q1_fusion_adaptation" / "phase3q1_training_manifest.json"
SEED = 20261004
ROUTES = ("S2", "SAR", "JOINT")


def emit(message: str) -> None:
    print(message, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as log:
        log.write(message + "\n")


def dump(name: str, value: object) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(row: dict) -> list[str]:
    return ["yes", "no"] if row["task_type"] == "binary_qa" else [str(item["key"]).lower() for item in row["options"]]


def parse_generated(route: str, result: dict) -> str | None:
    return result.get("parsed_answer")


def load_panel() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("test_access_count") != 0:
        raise RuntimeError("validation manifest does not prove zero test access")
    rows = [dict(row) for row in manifest["selection"]["validation"] if row["task_type"] in {"binary_qa", "multiple_choice_qa"}]
    rows.sort(key=lambda row: (row["task_type"], row["record_id"]))
    if len(rows) != 20 or Counter(row["task_type"] for row in rows) != {"binary_qa": 10, "multiple_choice_qa": 10}:
        raise RuntimeError("unexpected validation panel composition")
    if any(row.get("split") != "validation" for row in rows):
        raise RuntimeError("panel contains a non-validation identity")
    return rows


def validation_samples(rows: list[dict]) -> dict[str, object]:
    settings = get_settings()
    ids = sorted({row["image_id"] for row in rows} | {row["shuffled_image_id"] for row in rows})
    table = parquet.read_table(settings.dataset_root / "metadata.parquet", columns=["patch_id", "s1_name", "split"], filters=[("patch_id", "in", ids)])
    pairs = {str(item["patch_id"]): item for item in table.to_pylist()}
    if set(pairs) != set(ids) or any(str(item["split"]) != "validation" for item in pairs.values()):
        raise RuntimeError("filtered metadata did not return exactly the validation panel identities")
    result = {}
    for patch_id in ids:
        item = pairs[patch_id]
        s1_id = str(item["s1_name"])
        s2_scene = patch_id.rsplit("_", 2)[0]
        s1_scene = s1_id.rsplit("_", 3)[0]
        optical = {band: settings.dataset_root / "BigEarthNet-S2" / s2_scene / patch_id / f"{patch_id}_{band}.tif" for band in OPTICAL_BANDS}
        sar = {band: settings.dataset_root / "BigEarthNet-S1" / s1_scene / s1_id / f"{s1_id}_{band}.tif" for band in SAR_BANDS}
        reference = settings.dataset_root / "Reference_Maps" / s2_scene / patch_id / f"{patch_id}_reference_map.tif"
        sample = Sample(patch_id, optical, sar, reference, "validation")
        result[patch_id] = load_sample(sample)
    return result


def visual(controller, route: str, prepared):
    if route == "S2":
        tensor = torch.from_numpy(_robust_channel_scale(prepared.optical).astype(np.float32, copy=False)).unsqueeze(0).to(controller.device)
        return controller.projector(tensor)
    if route == "SAR":
        tokens = controller.croma.infer_modality(sar=prepared.sar.astype(np.float32, copy=False))["SAR_encodings"]
        return controller.projector(tokens)
    tokens = controller.croma.infer(prepared.optical, prepared.sar)["joint_encodings"]
    return controller.projector(tokens)


def constrained(controller, route: str, row: dict, prepared: object) -> dict:
    with torch.inference_mode():
        tokens = visual(controller, route, prepared)
        scores = {}
        for candidate in candidates(row):
            batch = build_qwen_visual_token_batch(tokenizer=controller.tokenizer, model=controller.model, visual_tokens=tokens, questions=[row["question"]], answers=[candidate])
            scores[candidate] = float(target_sequence_log_probability(controller.model(**batch).logits, batch["labels"]))
    prediction = max(scores, key=scores.get)
    return {"prediction": prediction, "scores": scores, "token_l2": float(torch.linalg.vector_norm(tokens).detach().cpu())}


def generated(controller, route: str, row: dict, prepared: object) -> dict:
    if route == "S2":
        raw = controller.run(image_id=prepared.patch_id, split="validation", optical=prepared.optical, metadata=prepared.metadata, question=row["question"], task_type=row["task_type"], choices=row.get("options"))
        text, parsed = raw["answer"], parse_generated(route, raw)
    elif route == "SAR":
        raw = controller.run(image_id=prepared.patch_id, split="validation", sar=prepared.sar, metadata=prepared.metadata, question=row["question"], task_type=row["task_type"], choices=row.get("options"))
        text, parsed = raw["answer"], parse_generated(route, raw)
    else:
        raw = controller.run_satquery(task_type=row["task_type"], question=row["question"], s1=prepared.sar, s2=prepared.optical, patch_id=prepared.patch_id, s1_patch_id=prepared.patch_id, s2_patch_id=prepared.patch_id, spatial_metadata=prepared.metadata)
        text, parsed = raw["generated_response"], parse_generated(route, raw)
    return {"text": text, "parsed": parsed, "correct": parsed == row["answer"] if parsed is not None else False, "route_raw_status": raw.get("status"), "warnings": raw.get("warnings", [])}


def metric(records: list[dict], field: str) -> dict:
    groups = {}
    for task in ("binary_qa", "multiple_choice_qa"):
        subset = [record for record in records if record["task_type"] == task]
        parsed = [record for record in subset if record[field] is not None]
        correct = sum(record[field] == record["answer"] for record in parsed)
        groups[task] = {"total": len(subset), "parsed": len(parsed), "correct": correct, "accuracy": correct / len(parsed) if parsed else None, "accuracy_over_total": correct / len(subset) if subset else None}
    return groups


def make_controller(route: str):
    return {"S2": SingleImageVQAController, "SAR": SingleImageSARVQAController, "JOINT": SatQueryV1Controller}[route]()


def close(controller) -> None:
    if hasattr(controller, "close"):
        controller.close()
    else:
        controller.croma = controller.projector = controller.model = controller.tokenizer = None
    del controller
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("semantic-improvement artifact namespace already exists")
    OUT.mkdir(parents=True)
    rows = load_panel()
    dump("experiment_config.json", {"git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "branch": __import__("subprocess").check_output(["git", "branch", "--show-current"], text=True).strip(), "seed": SEED, "validation_record_ids": [row["record_id"] for row in rows], "task_counts": dict(Counter(row["task_type"] for row in rows)), "models": {"s2": "frozen Phase 3O projector + Qwen", "sar": "frozen Phase 3P projector + CROMA + Qwen", "joint": "frozen Phase 3Q joint projector + CROMA + Qwen"}, "planned_metrics": ["parsed_accuracy", "accuracy_over_total", "shuffle_decision_change", "visual_token_l2"], "test_access": "forbidden"})
    emit("[SETUP] fixed validation panel: binary=10 mcq=10 test_access=0")
    samples = validation_samples(rows)
    baseline, post, diagnostics, runtime = [], [], {}, {}
    diagnostic_rows = [row for row in rows if sum(item["task_type"] == row["task_type"] for item in rows[:rows.index(row)+1]) <= 3]
    for route in ROUTES:
        torch.cuda.reset_peak_memory_stats()
        controller = make_controller(route)
        started = time.perf_counter()
        emit(f"[ROUTE] loading={route}")
        controller.load()
        route_diagnostics = []
        for index, row in enumerate(rows, 1):
            prepared = samples[row["image_id"]]
            base = generated(controller, route, row, prepared)
            improved = constrained(controller, route, row, prepared)
            record = {"record_id": row["record_id"], "logical_sample_id": row["image_id"], "task_type": row["task_type"], "question": row["question"], "answer": row["answer"], "route": route, "generated": base, "constrained": improved}
            baseline.append({**record, "prediction": base["parsed"]})
            post.append({**record, "prediction": improved["prediction"]})
            emit(f"[{index:02d}/{len(rows)}] route={route} task={row['task_type']} generated_correct={base['correct']} constrained_correct={improved['prediction'] == row['answer']}")
            if row in diagnostic_rows:
                shuffled = samples[row["shuffled_image_id"]]
                correct_visual = visual(controller, route, prepared)
                if route == "JOINT":
                    joint_shuffled = type(prepared)(prepared.patch_id, prepared.optical, shuffled.sar, prepared.raw_optical, shuffled.raw_sar, prepared.reference, prepared.metadata)
                    shuffled_result = constrained(controller, route, row, joint_shuffled)
                    condition = "correct_s2_shuffled_sar"
                else:
                    shuffled_result = constrained(controller, route, row, shuffled)
                    condition = "shuffled_image"
                shuffled_visual = visual(controller, route, joint_shuffled if route == "JOINT" else shuffled)
                route_diagnostics.append({"record_id": row["record_id"], "task_type": row["task_type"], "condition": condition, "correct_prediction": improved["prediction"], "shuffled_prediction": shuffled_result["prediction"], "decision_changed": improved["prediction"] != shuffled_result["prediction"], "visual_token_l2": float(torch.linalg.vector_norm(correct_visual - shuffled_visual).detach().cpu())})
        runtime[route] = {"seconds": time.perf_counter() - started, "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}
        diagnostics[route] = route_diagnostics
        close(controller)
    baseline_metrics = {route: metric([row for row in baseline if row["route"] == route], "prediction") for route in ROUTES}
    post_metrics = {route: metric([row for row in post if row["route"] == route], "prediction") for route in ROUTES}
    dump("baseline_predictions.json", baseline); dump("baseline_metrics.json", baseline_metrics)
    dump("modality_diagnostics.json", diagnostics)
    error = {
        route: {
            task: {
                "generated_unparsed": sum(row["prediction"] is None for row in baseline if row["route"] == route and row["task_type"] == task),
                "generated_answers": dict(Counter(row["prediction"] for row in baseline if row["route"] == route and row["task_type"] == task)),
            }
            for task in ("binary_qa", "multiple_choice_qa")
        }
        for route in ROUTES
    }
    dump("error_analysis.json", error)
    dump("post_predictions.json", post); dump("post_metrics.json", post_metrics)
    comparison = {route: {task: {"baseline": baseline_metrics[route][task], "post": post_metrics[route][task], "delta_accuracy_over_total": (post_metrics[route][task]["accuracy_over_total"] or 0) - (baseline_metrics[route][task]["accuracy_over_total"] or 0)} for task in baseline_metrics[route]} for route in ROUTES}
    dump("baseline_vs_post.json", comparison); dump("runtime_metrics.json", runtime)
    summary = {"classification": "SEMANTIC_PERFORMANCE_PARTIALLY_IMPROVED" if any(value[task]["delta_accuracy_over_total"] > 0 for value in comparison.values() for task in value) else "NO_RELIABLE_SEMANTIC_GAIN", "best_intervention": "validation-wide constrained candidate likelihood", "training_justified": False, "training_reason": "bounded non-training intervention measured first; no training initiated without its diagnostics supporting a learning opportunity", "test_images_read": 0, "test_labels_read": 0, "test_inference_runs": 0, "test_metrics_computed": 0, "comparison": comparison, "runtime": runtime}
    dump("final_summary.json", summary)
    emit("SATQUERY_SEMANTIC_IMPROVEMENT_RUN_COMPLETE")
    emit(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
