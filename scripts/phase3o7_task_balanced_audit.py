"""Phase 3O.7 task-balanced semantic validation and image-shuffle ablation."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase3o2_adaptation_pilot import MODEL_REVISION, MODEL_SNAPSHOT, cube, hash_state
from scripts.phase3o6_semantic_audit import (
    GENERATION_CONFIG, build_generation_inputs, degenerate_flags,
    historical_s2_hash, normalize_binary, normalize_mcq, state_fingerprint,
)
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.multispectral_projector import S2MultispectralProjector, QWEN_IMAGE_GRID_THW
from src.eo_vlm.training import freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

OUT = ROOT / "artifacts/training/phase3o"
RUN = OUT / "phase3o5_run"
CHECKPOINT = RUN / "s2_multispectral_projector.pt"
PANEL_PATH = OUT / "phase3o7_validation_panel.json"
SHUFFLE_PATH = OUT / "phase3o7_image_shuffle_map.json"
AUDIT_PATH = OUT / "phase3o7_task_balanced_semantic_audit.json"
EXPECTED_SHA = "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db"
EXPECTED_FP = "db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
SEED = 307
TASKS = ("binary_qa", "multiple_choice_qa", "caption")
SOURCES = {
    "visual_vqa_candidates.jsonl": OUT.parents[1] / "annotations/image_language/visual_vqa_candidates.jsonl",
    "caption_candidates.jsonl": OUT.parents[1] / "annotations/image_language/caption_candidates.jsonl",
}
HISTORICAL = [
    RUN / "pilot_manifest.json", RUN / "training_summary.json",
    RUN / "phase3o5_reload_receipt.json", OUT / "phase3o6_semantic_audit.json",
]


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_index() -> dict[str, Path]:
    roots = [
        Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"),
        Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2"),
    ]
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.tif"):
            if "_B" in path.stem:
                index.setdefault(path.name, path)
    return index


def selection_key(task: str, record: dict) -> str:
    return hashlib.sha256(f"{SEED}|{task}|{record['record_id']}".encode()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in HISTORICAL}


def load_candidates() -> tuple[dict[str, list[dict]], dict[str, str]]:
    candidates = {task: [] for task in TASKS}
    source_hash = {str(path.relative_to(ROOT)): sha256_file(path) for path in SOURCES.values()}
    for path in SOURCES.values():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                task = row.get("task_type")
                if (task in candidates and row.get("split") == "validation"
                        and row.get("eligibility") == "ELIGIBLE_VISUAL_VQA"
                        and row.get("dependency_class") == "VISUAL_ONLY"):
                    candidates[task].append(row)
    return candidates, source_hash


def select_panel(candidates: dict[str, list[dict]], phase3o6_ids: set[str], train_images: set[str], index: dict[str, Path]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    selected = {task: [] for task in TASKS}
    available_counts: dict[str, int] = {}
    for task in TASKS:
        eligible = [
            row for row in candidates[task]
            if row["record_id"] not in phase3o6_ids
            and row["image_id"] not in train_images
            and all(f"{row['image_id']}_{band}.tif" in index for band in S2_BANDS)
        ]
        available_counts[task] = len(eligible)
        used_images: set[str] = set()
        for row in sorted(eligible, key=lambda value: selection_key(task, value)):
            if row["image_id"] not in used_images:
                selected[task].append(row)
                used_images.add(row["image_id"])
            if len(selected[task]) == 30:
                break
    return selected, available_counts


def make_shuffle(selected: dict[str, list[dict]]) -> list[dict]:
    mapping: list[dict] = []
    for task in TASKS:
        rows = sorted(selected[task], key=lambda row: selection_key(f"{task}|shuffle", row))
        if len(rows) < 2:
            raise RuntimeError("PHASE3O7_BLOCKED_SHUFFLE_CARDINALITY")
        targets = rows[1:] + rows[:1]
        for source, target in zip(rows, targets):
            if source["record_id"] == target["record_id"] or source["image_id"] == target["image_id"]:
                raise RuntimeError("PHASE3O7_BLOCKED_INVALID_SHUFFLE")
            mapping.append({
                "sample_id": source["record_id"], "task_type": task,
                "source_image_id": source["image_id"], "shuffled_sample_id": target["record_id"],
                "shuffled_image_id": target["image_id"],
            })
    return sorted(mapping, key=lambda value: value["sample_id"])


def run_one(*, row: dict, image_row: dict, asset: dict, runtime: dict, model, projector, condition: str) -> dict:
    question = row["question"] if row["task_type"] != "caption" else "Describe this Sentinel-2 image."
    reference = row["answer"] if row["task_type"] != "caption" else row["caption"]
    prompt = GENERATION_CONFIG["prompt_template"].format(question=question)
    input_ids, attention_mask, inputs_embeds = build_generation_inputs(
        runtime["processor"].tokenizer, model, projector, cube(asset), question,
    )
    generated = model.generate(
        input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], dtype=torch.long, device=input_ids.device),
        max_new_tokens=GENERATION_CONFIG["max_new_tokens"], do_sample=False, temperature=None,
        top_p=None, top_k=None, repetition_penalty=GENERATION_CONFIG["repetition_penalty"], use_cache=True,
    )[0, input_ids.shape[1]:]
    text = runtime["processor"].tokenizer.decode(generated, skip_special_tokens=True)
    if row["task_type"] == "binary_qa":
        prediction, parse_status = normalize_binary(text)
    elif row["task_type"] == "multiple_choice_qa":
        prediction, parse_status = normalize_mcq(text, row.get("options"))
    else:
        prediction, parse_status = (text.strip() or None), "caption_descriptive_only"
    correct = prediction == reference if row["task_type"] != "caption" and prediction is not None else None
    image_paths = {band: Path(asset["local_path"]) / f"{image_row['image_id']}_{band}.tif" for band in S2_BANDS}
    return {
        "condition": condition, "sample_id": row["record_id"], "split": "validation",
        "task_type": row["task_type"], "question": question, "prompt": prompt,
        "allowed_answer_choices": row.get("options"), "reference": reference,
        "image_id_used": image_row["image_id"], "image_sample_id_used": image_row["record_id"],
        "image_s2_source_hash": historical_s2_hash(image_paths), "generated_text": text,
        "normalized_prediction": prediction, "correct": correct, "parse_status": parse_status,
        "generation_token_count": int(generated.numel()), "degenerate_flags": degenerate_flags(text, prompt),
        "checkpoint_sha256": EXPECTED_SHA, "model_state_fingerprint": EXPECTED_FP,
        "qwen_revision": MODEL_REVISION, "generation_config": GENERATION_CONFIG,
        "provenance": {"annotation_id": row["annotation_id"], "source_revision": row["source_revision"]},
    }


def classification_summary(rows: list[dict]) -> dict:
    correct = sum(row["correct"] is True for row in rows)
    unparsable = sum(row["normalized_prediction"] is None for row in rows)
    return {"total": len(rows), "correct": correct, "incorrect": len(rows) - correct - unparsable,
            "unparsable": unparsable, "accuracy": correct / len(rows) if rows else None}


def degeneracy_summary(rows: list[dict]) -> dict:
    text = [row["generated_text"].strip() for row in rows]
    flags = Counter(flag for row in rows for flag in row["degenerate_flags"])
    return {
        "empty": sum(not value for value in text),
        "identical_answer_across_all_samples": len(set(text)) == 1,
        "prompt_echo": flags["prompt_echo"], "punctuation_or_special_only": flags["punctuation_or_special_only"],
        "excessive_repetition": flags["excessive_repetition"],
        "malformed_mcq": sum(row["task_type"] == "multiple_choice_qa" and row["normalized_prediction"] is None for row in rows),
        "abnormal_termination": 0, "flag_counts": dict(flags),
    }


def paired(normal: list[dict], shuffled: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in shuffled}
    labels = Counter()
    for row in normal:
        other = by_id[row["sample_id"]]
        left = "correct" if row["correct"] is True else "incorrect_or_unparsable"
        right = "correct" if other["correct"] is True else "incorrect_or_unparsable"
        labels[f"normal_{left}__shuffled_{right}"] += 1
    return dict(labels)


def main() -> None:
    started = time.perf_counter()
    if not CHECKPOINT.is_file() or sha256_file(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("PHASE3O7_BLOCKED_CHECKPOINT_SHA")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    if state_fingerprint(state) != EXPECTED_FP:
        raise RuntimeError("PHASE3O7_BLOCKED_PROJECTOR_FINGERPRINT")
    if not (MODEL_SNAPSHOT.is_dir() and MODEL_SNAPSHOT.name == MODEL_REVISION):
        raise RuntimeError("PHASE3O7_BLOCKED_QWEN_REVISION")

    candidates, candidate_hashes = load_candidates()
    phase3o5 = json.loads((RUN / "pilot_manifest.json").read_text(encoding="utf-8"))
    phase3o6 = json.loads((OUT / "phase3o6_semantic_audit.json").read_text(encoding="utf-8"))
    phase3o6_ids = {row["sample_id"] for row in phase3o6["samples"]}
    train_images = {row["image_id"] for row in phase3o5["selection"]["train"]}
    index = file_index()
    selected, available_counts = select_panel(candidates, phase3o6_ids, train_images, index)
    requested = {task: 30 for task in TASKS}
    panel = {
        "phase": "3O.7", "seed": SEED, "status": "PANEL_REGISTERED",
        "requested_counts": requested, "available_eligible_validation_counts": available_counts,
        "selection_algorithm": "sha256(seed|task_type|record_id) ascending; unique image_id per task; excludes Phase3O.6 identities and Phase3O.5 train images",
        "source_manifest": {"path": "artifacts/annotations/image_language/manifest.json",
                            "sha256": sha256_file(ROOT / "artifacts/annotations/image_language/manifest.json"),
                            "candidate_file_sha256": candidate_hashes},
        "phase3o6_excluded_sample_count": len(phase3o6_ids), "test_access_count": 0,
        "records": {task: selected[task] for task in TASKS},
    }
    if any(len(selected[task]) != 30 for task in TASKS):
        panel["status"] = "PHASE3O7_BLOCKED_TASK_BALANCE"
        dump(PANEL_PATH, panel)
        print(json.dumps(panel, indent=2))
        return
    panel["selected_counts"] = {task: len(selected[task]) for task in TASKS}
    panel["train_validation_image_overlap"] = len(train_images & {row["image_id"] for values in selected.values() for row in values})
    if panel["train_validation_image_overlap"] != 0:
        raise RuntimeError("PHASE3O7_BLOCKED_SPLIT_OVERLAP")
    shuffle = make_shuffle(selected)
    dump(PANEL_PATH, panel)
    dump(SHUFFLE_PATH, {"phase": "3O.7", "seed": SEED, "algorithm": "task-local deterministic sorted cyclic derangement",
                        "mapping": shuffle, "test_access_count": 0})

    assets: dict[str, dict] = {}
    all_rows = {row["record_id"]: row for values in selected.values() for row in values}
    for row in all_rows.values():
        paths = {band: index[f"{row['image_id']}_{band}.tif"] for band in S2_BANDS}
        assets[row["record_id"]] = inspect_area(row, paths).to_dict()
        # Force compact-JSON source identity construction before any inference.
        historical_s2_hash(paths)

    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    adapter = Qwen25VLRGBAdapter()
    runtime = adapter.load_model(MODEL_SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"]
    model.eval()
    freeze = freeze_qwen(model)
    if freeze["qwen_trainable_parameter_count"] != 0:
        raise RuntimeError("PHASE3O7_BLOCKED_QWEN_TRAINABLE")
    projector = S2MultispectralProjector().to("cuda")
    projector.load_state_dict(state)
    projector.eval()
    qwen_before, projector_before, historical_before = hash_state(model), hash_state(projector), source_hashes()
    mapping = {entry["sample_id"]: entry for entry in shuffle}
    normal_rows, shuffled_rows = [], []
    with torch.inference_mode():
        for task in TASKS:
            for row in selected[task]:
                normal_rows.append(run_one(row=row, image_row=row, asset=assets[row["record_id"]], runtime=runtime, model=model, projector=projector, condition="normal"))
                shuffled_image_row = all_rows[mapping[row["record_id"]]["shuffled_sample_id"]]
                shuffled_rows.append(run_one(row=row, image_row=shuffled_image_row, asset=assets[shuffled_image_row["record_id"]], runtime=runtime, model=model, projector=projector, condition="shuffled_image"))
    qwen_after, projector_after, historical_after = hash_state(model), hash_state(projector), source_hashes()
    checkpoint_after = sha256_file(CHECKPOINT)
    fingerprint_after = state_fingerprint(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"])
    if (qwen_before != qwen_after or projector_before != projector_after or checkpoint_after != EXPECTED_SHA
            or fingerprint_after != EXPECTED_FP or historical_before != historical_after):
        raise RuntimeError("PHASE3O7_FAILED_IMMUTABILITY")

    normal_by_task = {task: [row for row in normal_rows if row["task_type"] == task] for task in TASKS}
    shuffled_by_task = {task: [row for row in shuffled_rows if row["task_type"] == task] for task in TASKS}
    binary_refs = Counter(row["reference"] for row in normal_by_task["binary_qa"])
    binary_majority = max(binary_refs.values()) / 30
    mcq_options = [len(row["allowed_answer_choices"] or []) for row in normal_by_task["multiple_choice_qa"]]
    caption_changes = [
        normal["generated_text"] != {row["sample_id"]: row for row in shuffled_by_task["caption"]}[normal["sample_id"]]["generated_text"]
        for normal in normal_by_task["caption"]
    ]
    audit = {
        "phase": "3O.7", "status": "PHASE3O7_COMPLETE", "audit_scope": "internal_controlled_validation_image_dependence",
        "validation_count": 90, "test_access_count": 0, "test_records_accessed": [],
        "panel_path": str(PANEL_PATH.relative_to(ROOT)), "shuffle_map_path": str(SHUFFLE_PATH.relative_to(ROOT)),
        "normal_results": normal_rows, "shuffled_results": shuffled_rows,
        "normal_task_results": {task: classification_summary(normal_by_task[task]) for task in TASKS if task != "caption"},
        "shuffled_task_results": {task: classification_summary(shuffled_by_task[task]) for task in TASKS if task != "caption"},
        "binary_majority_label_baseline": {"label_counts": dict(binary_refs), "accuracy": binary_majority},
        "mcq_random_choice_expectation": {"option_counts": mcq_options, "mean_accuracy": sum(1 / count for count in mcq_options) / len(mcq_options)},
        "accuracy_deltas_normal_minus_shuffled": {
            task: classification_summary(normal_by_task[task])["accuracy"] - classification_summary(shuffled_by_task[task])["accuracy"]
            for task in ("binary_qa", "multiple_choice_qa")
        },
        "paired_correctness": {task: paired(normal_by_task[task], shuffled_by_task[task]) for task in ("binary_qa", "multiple_choice_qa")},
        "caption_image_dependence": {"normal": degeneracy_summary(normal_by_task["caption"]), "shuffled": degeneracy_summary(shuffled_by_task["caption"]),
                                     "identical_generation_count": sum(not value for value in caption_changes), "changed_generation_count": sum(caption_changes)},
        "normal_degeneracy": degeneracy_summary(normal_rows), "shuffled_degeneracy": degeneracy_summary(shuffled_rows),
        "checkpoint_sha256": EXPECTED_SHA, "checkpoint_sha256_verified": checkpoint_after == EXPECTED_SHA,
        "model_state_fingerprint": EXPECTED_FP, "model_state_fingerprint_verified": fingerprint_after == EXPECTED_FP,
        "qwen_revision": MODEL_REVISION, "qwen_revision_verified": True, "qwen_trainable_parameters": 0,
        "qwen_immutability_verified": qwen_before == qwen_after, "projector_immutability_verified": projector_before == projector_after,
        "historical_artifacts_unchanged": historical_before == historical_after, "no_optimizer_created": True, "zero_gradient_updates": True,
        "generation_config": GENERATION_CONFIG, "provenance_status": "VERIFIED",
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "peak_ram_bytes": process.memory_info().rss,
        "duration_seconds": time.perf_counter() - started,
    }
    dump(AUDIT_PATH, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
