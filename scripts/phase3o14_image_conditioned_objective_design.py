"""Preregistered, projector-only Phase 3O.14 objective smoke test."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.phase3o11_postfix_semantic_audit import cube, hash_state
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.image_conditioned_objective import image_conditioned_loss, target_sequence_log_probability
from src.eo_vlm.multispectral_projector import QWEN_IMAGE_GRID_THW, S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.training import build_qwen_token_batch, freeze_qwen, optimizer_parameter_ids
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

RUN12 = ROOT / "artifacts/training/phase3o/phase3o12_run"
OUT = ROOT / "artifacts/training/phase3o/phase3o14_objective_design"
CK = RUN12 / "phase3o12_projector_final.pt"
SHA = "5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0"
FP = "168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b"
REV = "66285546d2b821cf421d4f5eb2576359d3770cd3"
SNAPSHOT = Path(r"C:\Users\Rishab\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots") / REV
TASKS = ("binary_qa", "multiple_choice_qa", "caption")
PREREG = {"phase": "3O.14", "seed": 3014, "initialization": "phase3o12_final_projector", "why_initialization": "diagnose whether the new objective can increase correct-versus-shuffled discrimination from the existing learned state", "base_lm_loss_weight": 1.0, "contrastive_loss_weight": 0.25, "margin": 0.25, "task_weights": {task: 1.0 for task in TASKS}, "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001, "smoke_steps": 4, "train_panel_counts": {task: 4 for task in TASKS}, "validation_micro_counts": {task: 2 for task in TASKS}, "test_access_count": 0,
          "objective": "L = L_CE(correct) + 0.25 * mean(max(0, 0.25 - (logP(target|correct,prompt) - logP(target|shuffled,prompt))))", "target_score": "sum of causal log probabilities over every non--100 label token, including EOS when supplied by existing supervision"}


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(f"PHASE3O14_BLOCKED: {message}")


def fingerprint(state: dict) -> str:
    projector = S2MultispectralProjector(); projector.load_state_dict(state)
    return projector.fingerprint()


def image_index() -> dict[str, Path]:
    paths = {}
    for root in (Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"), Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2")):
        for path in root.rglob("*.tif"):
            if "_B" in path.stem:
                paths.setdefault(path.name, path)
    return paths


def answer(row: dict) -> str:
    return row["caption"] if row["task_type"] == "caption" else row["answer"]


def question(row: dict) -> str:
    return "Describe this Sentinel-2 image." if row["task_type"] == "caption" else row["question"]


def asset(row: dict, paths: dict[str, Path], cache: dict) -> dict:
    if row["record_id"] not in cache:
        cache[row["record_id"]] = inspect_area(row, {band: paths[f"{row['image_id']}_{band}.tif"] for band in S2_BANDS}).to_dict()
    return cache[row["record_id"]]


def pair_mapping(train_rows: list[dict]) -> dict[str, dict]:
    mapping = {}
    for task in TASKS:
        rows = sorted((row for row in train_rows if row["task_type"] == task), key=lambda row: (row["image_id"], row["record_id"]))
        for index, row in enumerate(rows):
            peer = next(candidate for candidate in rows[index + 1:] + rows[:index + 1] if candidate["image_id"] != row["image_id"])
            mapping[row["record_id"]] = {"sample_id": row["record_id"], "task_type": task, "source_image_id": row["image_id"], "shuffled_sample_id": peer["record_id"], "shuffled_image_id": peer["image_id"], "split": "train"}
    return mapping


def score_row(model, tokenizer, projector, row, shuffled, paths, cache, *, require_grad: bool = False):
    correct = build_qwen_token_batch(tokenizer=tokenizer, model=model, projector=projector, s2=cube(asset(row, paths, cache)), questions=[question(row)], answers=[answer(row)])
    wrong = build_qwen_token_batch(tokenizer=tokenizer, model=model, projector=projector, s2=cube(asset(shuffled, paths, cache)), questions=[question(row)], answers=[answer(row)])
    correct_output = model(**correct); shuffled_output = model(**wrong)
    if require_grad:
        objective = image_conditioned_loss(correct_output.logits, shuffled_output.logits, correct["labels"], base_lm_loss=correct_output.loss, contrastive_weight=PREREG["contrastive_loss_weight"], margin=PREREG["margin"])
        return objective
    correct_score = target_sequence_log_probability(correct_output.logits, correct["labels"])
    shuffled_score = target_sequence_log_probability(shuffled_output.logits, wrong["labels"])
    return {"correct_score": float(correct_score.item()), "shuffled_score": float(shuffled_score.item()), "margin": float((correct_score - shuffled_score).item())}


def audit(model, tokenizer, projector, rows, mapping, lookup, paths, cache) -> dict:
    entries = []
    with torch.inference_mode():
        for row in rows:
            entries.append({"sample_id": row["record_id"], "task_type": row["task_type"], **score_row(model, tokenizer, projector, row, lookup[mapping[row["record_id"]]["shuffled_sample_id"]], paths, cache)})
    means = {task: sum(entry["margin"] for entry in entries if entry["task_type"] == task) / sum(entry["task_type"] == task for entry in entries) for task in TASKS}
    return {"entries": entries, "task_mean_margins": means}


def generate_caption(model, tokenizer, projector, row, paths, cache) -> str:
    batch = build_qwen_token_batch(tokenizer=tokenizer, model=model, projector=projector, s2=cube(asset(row, paths, cache)), questions=[question(row)], answers=None)
    result = generate_with_multimodal_prefix(model, input_ids=batch["input_ids"], inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"], image_grid_thw=batch["image_grid_thw"], max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, use_cache=True)
    return tokenizer.decode(result[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    print("PHASE3O14 START", flush=True)
    print(f"OBJECTIVE DEFINITION = {PREREG['objective']}", flush=True)
    require(not OUT.exists(), "namespace already exists")
    require(sha256_file(CK) == SHA, "Phase 3O.12 checkpoint SHA")
    initial_state = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
    require(fingerprint(initial_state) == FP and SNAPSHOT.is_dir(), "model identity")
    manifest = json.loads((RUN12 / "phase3o12_training_manifest.json").read_text())
    require(manifest["test_access_count"] == 0, "TEST access")
    train, validation = manifest["selection"]["train"], manifest["selection"]["validation"]
    require(all(row["split"] == "train" for row in train) and all(row["split"] == "validation" for row in validation), "partition integrity")
    mapping = pair_mapping(train); require(len(mapping) == 600 and all(entry["source_image_id"] != entry["shuffled_image_id"] for entry in mapping.values()), "train shuffle map")
    train_panel = [row for task in TASKS for row in sorted((item for item in train if item["task_type"] == task), key=lambda item: item["record_id"])[:4]]
    validation_panel = [row for task in TASKS for row in sorted((item for item in validation if item["task_type"] == task), key=lambda item: item["record_id"])[:2]]
    # Validation negatives are deterministically selected only from validation; no validation row is trained on.
    validation_mapping = pair_mapping([{**row, "split": "train"} for row in validation])
    OUT.mkdir(parents=True)
    dump(OUT / "phase3o14_objective_preregistration.json", PREREG)
    dump(OUT / "phase3o14_train_shuffle_map.json", {"phase": "3O.14", "seed": PREREG["seed"], "test_access_count": 0, "mapping": list(mapping.values())})
    paths, cache = image_index(), {}; lookup = {row["record_id"]: row for row in train}; validation_lookup = {row["record_id"]: row for row in validation}
    runtime = Qwen25VLRGBAdapter().load_model(SNAPSHOT, dtype="float16", device="cuda"); model = runtime["model"].eval(); tokenizer = runtime["processor"].tokenizer
    require(freeze_qwen(model)["qwen_trainable_parameter_count"] == 0, "Qwen trainability")
    projector = S2MultispectralProjector().to("cuda"); projector.load_state_dict(initial_state); projector.train()
    q_before, p_before = hash_state(model), hash_state(projector)
    pre = audit(model, tokenizer, projector, train_panel, mapping, lookup, paths, cache)
    validation_pre = audit(model, tokenizer, projector, validation_panel, validation_mapping, validation_lookup, paths, cache)
    dump(OUT / "phase3o14_pre_margin_audit.json", {"initialization": "phase3o12_final_projector", "train": pre, "validation_micro": validation_pre, "test_access_count": 0})
    print("BINARY PRE MARGIN =", pre["task_mean_margins"]["binary_qa"], flush=True); print("MCQ PRE MARGIN =", pre["task_mean_margins"]["multiple_choice_qa"], flush=True); print("CAPTION PRE MARGIN =", pre["task_mean_margins"]["caption"], flush=True)
    captions_pre = [generate_caption(model, tokenizer, projector, row, paths, cache) for row in train_panel if row["task_type"] == "caption"]
    optimizer = torch.optim.AdamW(projector.parameters(), lr=PREREG["learning_rate"], weight_decay=PREREG["weight_decay"])
    require(not optimizer_parameter_ids(optimizer).intersection(id(parameter) for parameter in model.parameters()), "Qwen optimizer membership")
    steps = []
    for step in range(1, PREREG["smoke_steps"] + 1):
        optimizer.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        for row in train_panel:
            objective = score_row(model, tokenizer, projector, row, lookup[mapping[row["record_id"]]["shuffled_sample_id"]], paths, cache, require_grad=True)
            weighted = objective["total_loss"] / len(train_panel)
            require(torch.isfinite(weighted), "nonfinite loss")
            # Accumulate only projector gradients; releasing each row's Qwen activation graph
            # keeps this deliberately tiny smoke test within the fixed 8 GB environment.
            weighted.backward()
            for name, value in objective.items(): totals[name] += float(value.detach()) / len(train_panel)
        gradients = [parameter.grad for parameter in projector.parameters()]; grad_norm = math.sqrt(sum(float(gradient.detach().float().pow(2).sum()) for gradient in gradients if gradient is not None))
        qwen_gradients = sum(parameter.grad is not None for parameter in model.parameters())
        require(grad_norm > 0 and qwen_gradients == 0, "gradient integrity"); optimizer.step()
        step_record = {"step": step, **dict(totals), "grad_norm": grad_norm, "qwen_gradient_tensors": qwen_gradients}
        steps.append(step_record)
        print(f"[3O14][SMOKE] step {step}/{PREREG['smoke_steps']}\nbase_loss = {totals['base_lm_loss']}\ncontrastive_loss = {totals['contrastive_loss']}\ntotal_loss = {totals['total_loss']}\ncorrect_score = {totals['correct_score']}\nshuffled_score = {totals['shuffled_score']}\nmargin = {totals['score_margin']}\ngrad_norm = {grad_norm}", flush=True)
    projector.eval(); post = audit(model, tokenizer, projector, train_panel, mapping, lookup, paths, cache); validation_post = audit(model, tokenizer, projector, validation_panel, validation_mapping, validation_lookup, paths, cache)
    captions_post = [generate_caption(model, tokenizer, projector, row, paths, cache) for row in train_panel if row["task_type"] == "caption"]
    changes = {task: post["task_mean_margins"][task] - pre["task_mean_margins"][task] for task in TASKS}
    print("BINARY POST MARGIN =", post["task_mean_margins"]["binary_qa"], flush=True); print("MCQ POST MARGIN =", post["task_mean_margins"]["multiple_choice_qa"], flush=True); print("CAPTION POST MARGIN =", post["task_mean_margins"]["caption"], flush=True); print("MARGIN CHANGE =", changes, flush=True); print("QWEN GRADIENTS = 0\nTEST ACCESS = 0", flush=True)
    smoke_path = OUT / "phase3o14_projector_smoke.pt"; torch.save({"state_dict": projector.state_dict(), "parent_checkpoint_sha256": SHA, "preregistration": PREREG}, smoke_path)
    artifact = {"phase": "3O.14", "status": "PHASE3O14_COMPLETE", "preregistration": PREREG, "test_access_count": 0, "checkpoint_parent": {"sha256": SHA, "fingerprint": FP}, "train_panel": [row["record_id"] for row in train_panel], "validation_micro_panel": [row["record_id"] for row in validation_panel], "steps": steps, "pre": pre, "post": post, "validation_pre": validation_pre, "validation_post": validation_post, "train_margin_change": changes, "validation_margin_change": {task: validation_post["task_mean_margins"][task] - validation_pre["task_mean_margins"][task] for task in TASKS}, "caption_micro": {"references": [answer(row) for row in train_panel if row["task_type"] == "caption"], "pre_generations": captions_pre, "post_generations": captions_post, "pre_unique": len(set(captions_pre)), "post_unique": len(set(captions_post))}, "projector_gradient_nonzero": all(step["grad_norm"] > 0 for step in steps), "qwen_gradient_tensors": 0, "qwen_unchanged": hash_state(model) == q_before, "phase3o12_projector_unchanged": sha256_file(CK) == SHA and fingerprint(torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]) == FP, "smoke_checkpoint": str(smoke_path.relative_to(ROOT)), "classification": "OBJECTIVE_PROMISING" if all(value > 0 for value in changes.values()) else "OBJECTIVE_MOVES_TRAIN_MARGIN_ONLY" if sum(value > 0 for value in changes.values()) >= 2 else "OBJECTIVE_NO_EFFECT"}
    require(artifact["qwen_unchanged"] and artifact["phase3o12_projector_unchanged"], "post-smoke immutability")
    dump(OUT / "phase3o14_smoke_training.json", artifact)
    dump(OUT / "phase3o14_post_margin_audit.json", {"train": post, "validation_micro": validation_post, "test_access_count": 0, "classification": artifact["classification"]})
    print("OBJECTIVE CLASSIFICATION =", artifact["classification"], flush=True); print("RECOMMENDED NEXT PHASE = Phase 3O.15 controlled full training only if separately authorized", flush=True)


if __name__ == "__main__":
    main()
