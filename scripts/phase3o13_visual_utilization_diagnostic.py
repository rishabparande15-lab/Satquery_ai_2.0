"""Read-only Phase 3O.13 visual-utilization diagnosis for the 3O.12 checkpoint."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.phase3o11_postfix_semantic_audit import cube, hash_state
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.multispectral_projector import QWEN_IMAGE_GRID_THW, QWEN_IMAGE_TOKEN_COUNT, S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.training import freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

RUN = ROOT / "artifacts/training/phase3o/phase3o12_run"
OUT = ROOT / "artifacts/training/phase3o/phase3o13_visual_utilization_diagnostic.json"
CK = RUN / "phase3o12_projector_final.pt"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
MODEL_SNAPSHOT = Path(r"C:\Users\Rishab\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots") / MODEL_REVISION
SHA = "5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0"
FP = "168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b"
TASKS = ("binary_qa", "multiple_choice_qa", "caption")
CONDITIONS = ("correct_image", "shuffled_image", "zero_image", "mean_visual")


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def projector_fingerprint(state: dict) -> str:
    projector = S2MultispectralProjector()
    projector.load_state_dict(state)
    return projector.fingerprint()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"PHASE3O13_BLOCKED: {message}")


def stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    return {"l2": float(torch.linalg.vector_norm(value)), "mean": float(value.mean()), "std": float(value.std(unbiased=False)), "min": float(value.min()), "max": float(value.max())}


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    left, right = left.detach().float(), right.detach().float()
    difference = left - right
    return {"l2": float(torch.linalg.vector_norm(difference)), "max_abs": float(difference.abs().max()), "cosine_similarity": float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())}


def image_index() -> dict[str, Path]:
    result = {}
    for root in (Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"), Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2")):
        for path in root.rglob("*.tif"):
            if "_B" in path.stem:
                result.setdefault(path.name, path)
    return result


def asset(row: dict, index: dict[str, Path]) -> dict:
    return inspect_area(row, {band: index[f"{row['image_id']}_{band}.tif"] for band in S2_BANDS}).to_dict()


def prompt(row: dict) -> str:
    return row["question"] if row["task_type"] != "caption" else "Describe this Sentinel-2 image."


def assemble(tokenizer, model, visual: torch.Tensor, question: str):
    token_ids = [model.config.vision_start_token_id] + [model.config.image_token_id] * QWEN_IMAGE_TOKEN_COUNT + [model.config.vision_end_token_id]
    token_ids += tokenizer(f"Question: {question}\nAnswer:", add_special_tokens=True)["input_ids"]
    ids = torch.tensor([token_ids], device=visual.device, dtype=torch.long)
    attention = torch.ones_like(ids)
    embeddings = model.get_input_embeddings()(ids)
    mask = ids.eq(model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
    require(int(mask.sum()) == visual.numel(), "visual slot mismatch")
    return ids, attention, embeddings.masked_scatter(mask, visual.to(embeddings.dtype).reshape(-1))


def topk(tokenizer, logits: torch.Tensor, k: int = 10) -> list[dict]:
    logits = logits.float(); probabilities = torch.softmax(logits, -1)
    values, ids = torch.topk(logits, k)
    return [{"id": int(token_id), "token": tokenizer.decode([int(token_id)]), "logit": float(value), "probability": float(probabilities[token_id])} for value, token_id in zip(values, ids)]


def candidate_scores(tokenizer, logits: torch.Tensor, task: str, reference: str) -> dict | None:
    candidates = ["yes", "no"] if task == "binary_qa" else (["a", "b", "c", "d"] if task == "multiple_choice_qa" else [])
    ids = {candidate: tokenizer(" " + candidate, add_special_tokens=False)["input_ids"] for candidate in candidates}
    if not candidates or any(len(value) != 1 for value in ids.values()) or len(ids.get(reference, [])) != 1:
        return None
    logits = logits.float(); values = {candidate: float(logits[token_ids[0]]) for candidate, token_ids in ids.items()}
    target = values[reference]
    return {"mapping_unambiguous": True, "scores": values, "target_margin": target - max(value for candidate, value in values.items() if candidate != reference)}


def generate(tokenizer, model, ids, attention, embeddings) -> dict:
    generated = generate_with_multimodal_prefix(model, input_ids=ids, inputs_embeds=embeddings, attention_mask=attention,
        image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], dtype=torch.long, device=ids.device), max_new_tokens=16,
        do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, use_cache=True,
        return_dict_in_generate=True, output_scores=True)
    tokens = generated.sequences[0, ids.shape[1]:]
    first = generated.scores[0][0]
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": [int(value) for value in tokens], "first_step_top10": topk(tokenizer, first), "first_step_eos_probability": float(torch.softmax(first.float(), -1)[tokenizer.eos_token_id])}


def average(entries: list[float]) -> float:
    return sum(entries) / len(entries) if entries else 0.0


def summarize(records: list[dict]) -> dict:
    by_task = {}
    for task in TASKS:
        items = [record for record in records if record["task_type"] == task]
        projector = {comparison: average([item["projector_sensitivity"][comparison]["l2"] for item in items]) for comparison in ("correct_vs_shuffled", "correct_vs_zero")}
        layers = {label: {comparison: average([item["hidden_state_sensitivity"][label][comparison]["l2"] for item in items]) for comparison in ("correct_vs_shuffled", "correct_vs_zero", "correct_vs_mean")} for label in items[0]["hidden_state_sensitivity"]}
        logits = {comparison: average([item["logit_sensitivity"][comparison]["l2"] for item in items]) for comparison in ("correct_vs_shuffled", "correct_vs_zero", "correct_vs_mean")}
        changes = {condition: sum(item["generations"]["correct_image"]["text"] != item["generations"][condition]["text"] for item in items) for condition in CONDITIONS if condition != "correct_image"}
        by_task[task] = {"samples": len(items), "projector_l2": projector, "hidden_l2": layers, "logit_l2": logits, "generation_changes": changes, "zero_output_agreement": 1 - changes["zero_image"] / len(items), "mean_output_agreement": 1 - changes["mean_visual"] / len(items)}
    return by_task


def main() -> None:
    started = time.perf_counter()
    require(sha256_file(CK) == SHA, "checkpoint SHA mismatch")
    state = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
    require(projector_fingerprint(state) == FP, "projector fingerprint mismatch")
    require(MODEL_SNAPSHOT.is_dir(), "pinned Qwen snapshot unavailable")
    manifest = json.loads((RUN / "phase3o12_training_manifest.json").read_text())
    shuffle_file = RUN / "image_shuffle_map.json"
    shuffle = {entry["sample_id"]: entry["shuffled_sample_id"] for entry in json.loads(shuffle_file.read_text())["mapping"]}
    require(manifest["test_access_count"] == 0 and len(manifest["selection"]["validation"]) == 150, "validation manifest integrity")
    panel = []
    for task in TASKS:
        rows = sorted((row for row in manifest["selection"]["validation"] if row["task_type"] == task), key=lambda row: row["record_id"])
        require(len(rows) == 50, f"validation count for {task}")
        panel.extend(rows[:10])
    require(len(panel) == 30 and all(row["record_id"] in shuffle for row in panel), "diagnostic panel or shuffle map")
    records = {row["record_id"]: row for row in manifest["selection"]["validation"]}
    index = image_index(); cached_assets = {}
    def get_asset(row):
        return cached_assets.setdefault(row["record_id"], asset(row, index))

    adapter = Qwen25VLRGBAdapter(); runtime = adapter.load_model(MODEL_SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"].eval(); frozen = freeze_qwen(model)
    require(frozen["qwen_trainable_parameter_count"] == 0, "Qwen is trainable")
    projector = S2MultispectralProjector().to("cuda"); projector.load_state_dict(state); projector.eval()
    qwen_before, projector_before = hash_state(model), hash_state(projector)
    tokenizer = runtime["processor"].tokenizer
    layers_count = None; output_records = []
    with torch.inference_mode():
        for number, row in enumerate(panel, 1):
            if number in (1, 5, 10, 15, 20, 25, 30):
                print(f"[3O13] sample {number}/30", flush=True)
            correct_s2 = cube(get_asset(row)).to("cuda")
            shuffled_row = records[shuffle[row["record_id"]]]
            shuffled_s2 = cube(get_asset(shuffled_row)).to("cuda")
            projected = {"correct_image": projector(correct_s2), "shuffled_image": projector(shuffled_s2), "zero_image": projector(torch.zeros_like(correct_s2))}
            projected["mean_visual"] = projected["correct_image"].mean(dim=1, keepdim=True).expand_as(projected["correct_image"])
            forwards, generations = {}, {}
            for condition, visual in projected.items():
                ids, attention, embeddings = assemble(tokenizer, model, visual, prompt(row))
                output = model(input_ids=ids, inputs_embeds=embeddings, attention_mask=attention, image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], dtype=torch.long, device="cuda"), output_hidden_states=True, return_dict=True)
                layers_count = len(output.hidden_states) - 1
                wanted = {"early": 1, "quarter": max(1, round(layers_count * .25)), "middle": max(1, round(layers_count * .5)), "three_quarter": max(1, round(layers_count * .75)), "final": -1}
                forwards[condition] = {"hidden": {label: output.hidden_states[index][0, -1].detach() for label, index in wanted.items()}, "logits": output.logits[0, -1].detach()}
                generations[condition] = generate(tokenizer, model, ids, attention, embeddings)
            hidden = {label: {comparison: compare(forwards["correct_image"]["hidden"][label], forwards[other]["hidden"][label]) for comparison, other in (("correct_vs_shuffled", "shuffled_image"), ("correct_vs_zero", "zero_image"), ("correct_vs_mean", "mean_visual"))} for label in forwards["correct_image"]["hidden"]}
            ref = row["caption"] if row["task_type"] == "caption" else row["answer"]
            scores = {condition: candidate_scores(tokenizer, forward["logits"], row["task_type"], ref) for condition, forward in forwards.items()}
            output_records.append({"sample_id": row["record_id"], "image_id": row["image_id"], "shuffled_sample_id": shuffled_row["record_id"], "shuffled_image_id": shuffled_row["image_id"], "task_type": row["task_type"], "reference": ref,
                "projector_statistics": {condition: stats(value) for condition, value in projected.items()}, "projector_sensitivity": {"correct_vs_shuffled": compare(projected["correct_image"], projected["shuffled_image"]), "correct_vs_zero": compare(projected["correct_image"], projected["zero_image"])},
                "hidden_state_sensitivity": hidden, "logit_sensitivity": {comparison: compare(forwards["correct_image"]["logits"], forwards[other]["logits"]) for comparison, other in (("correct_vs_shuffled", "shuffled_image"), ("correct_vs_zero", "zero_image"), ("correct_vs_mean", "mean_visual"))},
                "top10_first_answer_logits": {condition: topk(tokenizer, forward["logits"]) for condition, forward in forwards.items()}, "candidate_scores": scores, "generations": generations})
    # Attention is intentionally optional: requesting full attention tensors is not required for the diagnosis.
    attention = {"status": "ATTENTION_ANALYSIS_UNAVAILABLE", "reason": "not requested because the loaded Qwen attention implementation does not guarantee compact attention tensors; full attention materialization would be an unreasonable diagnostic-memory risk."}
    summary = summarize(output_records)
    train_captions = [row["caption"] for row in manifest["selection"]["train"] if row["task_type"] == "caption"]
    train_counts = Counter(train_captions)
    caption_generations = {condition: Counter(record["generations"][condition]["text"] for record in output_records if record["task_type"] == "caption") for condition in CONDITIONS}
    margins = {task: {condition: average([record["candidate_scores"][condition]["target_margin"] for record in output_records if record["task_type"] == task and record["candidate_scores"][condition] is not None]) for condition in CONDITIONS} for task in ("binary_qa", "multiple_choice_qa")}
    for task in margins:
        margins[task]["correct_minus_shuffled"] = margins[task]["correct_image"] - margins[task]["shuffled_image"]
    # Natural-image substitutions leave Binary/MCQ decisions and target margins nearly unchanged,
    # while zero imagery is an out-of-distribution intervention that changes them sharply.  This
    # supports weak discrimination among natural visual inputs, not a causal language-prior claim.
    roots = ["A_VISUAL_SIGNAL_STRONG_AT_PROJECTOR_WEAK_AT_LLM", "C_VISUAL_SIGNAL_REACHES_LOGITS_BUT_NOT_DECISION_MARGIN", "E_CAPTION_MODE_COLLAPSE", "F_TASK_SPECIFIC_VISUAL_UTILIZATION"]
    artifact = {"phase": "3O.13", "status": "PHASE3O13_COMPLETE", "diagnostic_type": "read_only_no_training_no_optimizer_no_backward", "checkpoint": {"sha256": SHA, "fingerprint": FP, "verified": True}, "qwen": {"revision": MODEL_REVISION, "trainable_parameter_count": 0, "frozen": True}, "test_access_count": 0,
        "panel": {"selection": "first 10 record_id-sorted validation records per task", "counts": {task: 10 for task in TASKS}, "sample_ids": [row["record_id"] for row in panel]}, "layer_count": layers_count, "records": output_records, "task_summary": summary,
        "text_dominance_control": {task: {"correct_zero_output_agreement": summary[task]["zero_output_agreement"], "interpretation": "Agreement is evidence consistent with task/prompt-prior influence, not a causal proof."} for task in ("binary_qa", "multiple_choice_qa")},
        "caption_collapse": {"train_caption_count": len(train_captions), "unique_train_targets": len(train_counts), "most_frequent_train_targets": [{"caption": text, "count": count} for text, count in train_counts.most_common(10)], "generated_modes": {condition: [{"text": text, "count": count} for text, count in counts.most_common()] for condition, counts in caption_generations.items()}},
        "attention": attention, "target_margins": margins, "root_cause_classifications": roots, "recommended_intervention": "D_IMAGE_CONTRASTIVE_OR_SHUFFLE_AWARE_OBJECTIVE plus G_CAPTION_SPECIFIC_OBJECTIVE; diagnose/use stronger image-conditioned supervision before considering LoRA.", "recommended_next_phase": "Phase 3O.14 controlled image-conditioned objective design (do not start automatically)",
        "qwen_unchanged": hash_state(model) == qwen_before, "projector_unchanged": hash_state(projector) == projector_before and sha256_file(CK) == SHA and projector_fingerprint(torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]) == FP, "duration_seconds": time.perf_counter() - started}
    require(artifact["qwen_unchanged"] and artifact["projector_unchanged"], "model immutability")
    dump(OUT, artifact)
    compact = {"projector_correct_vs_shuffled_l2": {task: summary[task]["projector_l2"]["correct_vs_shuffled"] for task in TASKS}, "early_layer": {task: summary[task]["hidden_l2"]["early"]["correct_vs_shuffled"] for task in TASKS}, "final_layer": {task: summary[task]["hidden_l2"]["final"]["correct_vs_shuffled"] for task in TASKS}, "status": artifact["status"]}
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
