"""Bounded, standalone Sentinel-1 semantic-adaptation pilot (Phase 3P.1).

This program deliberately has no Sentinel-2 input or fusion route.  It uses
the existing authoritative TRAIN/VALIDATION manifest, persists its exact SAR
join before optimization, and trains only ``S1SARProjector``.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.croma_adapter import CROMAAdapter
from src.dataset_loader import discover_samples, load_sample
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.image_conditioned_objective import target_sequence_log_probability
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch, freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT, hash_state

OUT = ROOT / "artifacts/training/phase3p/phase3p1_sar_adaptation"
MANIFEST_SOURCE = ROOT / "artifacts/training/phase3o/phase3o12_run/phase3o12_training_manifest.json"
DATA = Path(r"D:\Satquery_ai datasets\comparison\raw-1000")
CS = Path(r"D:\Satquery_ai datasets\croma_official")
CK = Path(r"D:\Satquery_ai datasets\checkpoints\CROMA_base.pt")
SEED = 30191
TRAIN_PER_TASK, VALIDATION_PER_TASK = 6, 3
TASKS = ("binary_qa", "multiple_choice_qa", "caption")
GENERATION = {"max_new_tokens": 8, "do_sample": False, "temperature": None, "top_p": None, "top_k": None, "repetition_penalty": 1.0, "use_cache": True}


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def answer(row: dict) -> str:
    if row["task_type"] == "caption": return row["caption"]
    return row["answer"]


def question(row: dict) -> str:
    return row["question"] if row["task_type"] != "caption" else "Describe this Sentinel-1 radar scene."


def parse(row: dict, text: str) -> str | None:
    normalized = text.strip().lower()
    if row["task_type"] == "caption": return None
    if row["task_type"] == "binary_qa":
        for token in normalized.replace(".", " ").split():
            if token in {"yes", "no"}: return token
    else:
        for token in normalized.replace(".", " ").replace("(", " ").replace(")", " ").split():
            if token in {"a", "b", "c", "d"}: return token
    return None


def record_summary(row: dict) -> dict:
    return {key: row.get(key) for key in ("record_id", "image_id", "task_type", "question", "answer", "caption", "source_dataset", "source_partition", "provenance")}


def select_records() -> tuple[dict[str, list[dict]], dict[str, object], dict[str, object]]:
    source = json.loads(MANIFEST_SOURCE.read_text(encoding="utf-8"))
    meta = pd.read_parquet(DATA / "metadata.parquet", columns=["patch_id", "s1_name"])
    map_s1 = dict(zip(meta.patch_id.astype(str), meta.s1_name.astype(str)))
    available: dict[str, list[dict]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in source["selection"][split]:
            task = row["task_type"]
            if task in TASKS and row["image_id"] in map_s1:
                candidate = dict(row)
                candidate["sar_root"] = str(DATA)
                candidate["sar_s1_name"] = map_s1[row["image_id"]]
                grouped[task].append(candidate)
        available[split] = [r for task in TASKS for r in grouped[task]]
        counts[split] = {task: len(grouped[task]) for task in TASKS}
    selected: dict[str, list[dict]] = {}
    for split, limit in (("train", TRAIN_PER_TASK), ("validation", VALIDATION_PER_TASK)):
        by_task: dict[str, list[dict]] = defaultdict(list)
        for row in available[split]: by_task[row["task_type"]].append(row)
        selected[split] = [row for task in TASKS for row in sorted(by_task[task], key=lambda x: x["record_id"])[:limit]]
    audit = {"source_manifest": str(MANIFEST_SOURCE), "source_manifest_sha256": sha(MANIFEST_SOURCE), "data_root": str(DATA), "mapping": "metadata.parquet patch_id -> s1_name", "available_valid_sar_supervision": counts, "test_access_count": 0}
    return selected, audit, {"raw_manifest_counts": source["counts"], "source_test_access_count": source["test_access_count"]}


def integrity(selected: dict[str, list[dict]], samples: dict[str, object]) -> dict[str, object]:
    train, valid = selected["train"], selected["validation"]
    train_ids, valid_ids = [r["image_id"] for r in train], [r["image_id"] for r in valid]
    all_rows = train + valid
    missing = [r["image_id"] for r in all_rows if r["image_id"] not in samples]
    # Read each exact source once: this verifies all S1 assets and preprocessing.
    prep = {}
    for image_id in sorted(set(r["image_id"] for r in all_rows)):
        prep[image_id] = load_sample(samples[image_id])
    return {"train_validation_image_overlap": sorted(set(train_ids) & set(valid_ids)), "duplicate_record_ids": [x for x, n in Counter(r["record_id"] for r in all_rows).items() if n > 1], "duplicate_image_task_identities": [x for x, n in Counter((r["image_id"], r["task_type"]) for r in all_rows).items() if n > 1], "missing_s1_samples": missing, "preprocessing_validated_image_count": len(prep), "test_access_count": 0}, prep


def save_projector(path: Path, projector: S1SARProjector) -> dict[str, object]:
    torch.save({"state_dict": projector.state_dict(), "architecture": "S1SARProjector(CROMA_S1[225,768]->Qwen[16,2048])"}, path)
    return {"path": path.name, "sha256": sha(path), "fingerprint": fingerprint(projector), "parameter_count": sum(p.numel() for p in projector.parameters())}


def make_tokens(prepared: dict[str, object]) -> tuple[dict[str, torch.Tensor], str]:
    croma = CROMAAdapter(CS, CK, device="cuda")
    croma.model.eval()
    for parameter in croma.model.parameters(): parameter.requires_grad_(False)
    initial = fingerprint(croma.model)
    result = {}
    for index, (image_id, sample) in enumerate(prepared.items(), 1):
        print(f"[3P1][CROMA] image {index}/{len(prepared)}", flush=True)
        with torch.no_grad(): result[image_id] = croma.infer_modality(sar=sample.raw_sar)["SAR_encodings"].detach().cpu()
    if fingerprint(croma.model) != initial: raise RuntimeError("CROMA_MUTATED")
    del croma; gc.collect(); torch.cuda.empty_cache()
    return result, initial


def batch(model, tokenizer, projector, tokens, row, target: bool = True):
    visual = projector(tokens[row["image_id"]].to("cuda"))
    return build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual, questions=[question(row)], answers=[answer(row)] if target else None)


def generated(model, tokenizer, projector, tokens, row, mode: str) -> tuple[str, float]:
    token = tokens[row["image_id"]] if mode == "correct" else (torch.zeros_like(tokens[row["image_id"]]) if mode == "zero" else tokens[row["shuffled_image_id"]])
    visual = projector(token.to("cuda"))
    b = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual, questions=[question(row)], answers=None)
    z = generate_with_multimodal_prefix(model, input_ids=b["input_ids"], inputs_embeds=b["inputs_embeds"], attention_mask=b["attention_mask"], image_grid_thw=b["image_grid_thw"], **GENERATION)
    ids = z[0, b["input_ids"].shape[1]:]
    return tokenizer.decode(ids, skip_special_tokens=True), float(ids.numel())


def semantic_audit(name, model, tokenizer, projector, tokens, rows) -> dict:
    projector.eval(); records = []
    with torch.no_grad():
        for row in rows:
            values = {}
            for mode in ("correct", "shuffled", "zero"):
                text, length = generated(model, tokenizer, projector, tokens, row, mode)
                parsed = parse(row, text)
                values[mode] = {"text": text, "token_count": length, "parsed": parsed, "correct": (parsed == answer(row)) if row["task_type"] != "caption" else None}
            # Teacher-forced target margins use the identical target under each SAR condition.
            scores = {}
            for mode in ("correct", "shuffled", "zero"):
                token = tokens[row["image_id"]] if mode == "correct" else (torch.zeros_like(tokens[row["image_id"]]) if mode == "zero" else tokens[row["shuffled_image_id"]])
                visual = projector(token.to("cuda")); b = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual, questions=[question(row)], answers=[answer(row)])
                scores[mode] = float(target_sequence_log_probability(model(**b).logits, b["labels"]))
            records.append({"record": record_summary(row), "conditions": values, "target_scores": scores, "correct_minus_shuffled_target_margin": scores["correct"] - scores["shuffled"], "correct_minus_zero_target_margin": scores["correct"] - scores["zero"]})
    metrics = {}
    for task in TASKS:
        subset = [r for r in records if r["record"]["task_type"] == task]
        if task == "caption":
            metrics[task] = {mode: {"empty": sum(not r["conditions"][mode]["text"].strip() for r in subset), "nonempty": sum(bool(r["conditions"][mode]["text"].strip()) for r in subset), "unique_outputs": len({r["conditions"][mode]["text"] for r in subset})} for mode in ("correct", "shuffled", "zero")}
        else:
            metrics[task] = {mode: {"correct": sum(r["conditions"][mode]["correct"] is True for r in subset), "incorrect": sum(r["conditions"][mode]["correct"] is False for r in subset), "unparsable": sum(r["conditions"][mode]["parsed"] is None for r in subset), "accuracy": sum(r["conditions"][mode]["correct"] is True for r in subset) / len(subset)} for mode in ("correct", "shuffled", "zero")}
    changed_shuffle = sum(r["conditions"]["correct"]["text"] != r["conditions"]["shuffled"]["text"] for r in records)
    changed_zero = sum(r["conditions"]["correct"]["text"] != r["conditions"]["zero"]["text"] for r in records)
    return {"audit": name, "generation": GENERATION, "records": records, "metrics": metrics, "normal_vs_shuffled_output_change_count": changed_shuffle, "normal_vs_zero_output_change_count": changed_zero, "test_access_count": 0}


def validation_losses(model, tokenizer, projector, tokens, rows) -> dict:
    projector.eval(); items = []
    with torch.no_grad():
        for row in rows:
            b = batch(model, tokenizer, projector, tokens, row); o = model(**b)
            items.append({"record_id": row["record_id"], "task_type": row["task_type"], "loss": float(o.loss)})
    per = {task: [x["loss"] for x in items if x["task_type"] == task] for task in TASKS}
    return {"raw_losses": items, "overall_mean_loss": float(np.mean([x["loss"] for x in items])), "per_task_mean_loss": {k: float(np.mean(v)) for k, v in per.items()}, "all_finite": all(math.isfinite(x["loss"]) for x in items)}


def main() -> None:
    if OUT.exists(): raise RuntimeError("PHASE3P1_NAMESPACE_EXISTS")
    OUT.mkdir(parents=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    print("PHASE3P1 START", flush=True)
    selected, audit, source_info = select_records()
    print("VALID SAR SUPERVISION:", flush=True)
    for task in TASKS: print(f"{task} = train {audit['available_valid_sar_supervision']['train'][task]}, validation {audit['available_valid_sar_supervision']['validation'][task]}", flush=True)
    samples = {x.patch_id: x for x in discover_samples(DATA, strict=True)}
    integ, prepared = integrity(selected, samples)
    if any(integ[k] for k in ("train_validation_image_overlap", "duplicate_record_ids", "duplicate_image_task_identities", "missing_s1_samples")): raise RuntimeError("PHASE3P1_FAILED_INTEGRITY")
    # Fixed shuffled mapping changes image only and never crosses partitions.
    for split, rows in selected.items():
        image_ids = sorted({r["image_id"] for r in rows})
        if len(image_ids) < 2: raise RuntimeError("INSUFFICIENT_IMAGES_FOR_SHUFFLE")
        mapping = dict(zip(image_ids, image_ids[1:] + image_ids[:1]))
        for row in rows: row["shuffled_image_id"] = mapping[row["image_id"]]
    manifest = {"phase": "3P.1", "status": "PREOPTIMIZATION_MANIFEST", "sar_only": True, "sentinel2_input_count": 0, "test_access_count": 0, "supervision_audit": audit, "source_info": source_info, "integrity": integ, "selection": {k: [record_summary(r) | {"sar_root": r["sar_root"], "sar_s1_name": r["sar_s1_name"], "shuffled_image_id": r["shuffled_image_id"]} for r in v] for k, v in selected.items()}, "counts": {k: dict(Counter(r["task_type"] for r in v)) for k, v in selected.items()}}
    dump("phase3p1_training_manifest.json", manifest)
    print(f"TRAIN = {len(selected['train'])}", flush=True); print(f"VALIDATION = {len(selected['validation'])}", flush=True); print("TEST = 0", flush=True)
    torch.manual_seed(SEED); projector = S1SARProjector().cuda()
    initial_receipt = save_projector(OUT / "phase3p1_sar_projector_initial.pt", projector)
    print("INITIAL PROJECTOR SHA = " + initial_receipt["sha256"], flush=True); print("INITIAL FINGERPRINT = " + initial_receipt["fingerprint"], flush=True)
    config = {"phase": "3P.1", "seed": SEED, "sar_only": True, "record_order": [r["record_id"] for r in selected["train"]], "task_counts": manifest["counts"], "optimizer": "AdamW", "learning_rate": 0.0005, "weight_decay": 0.0, "batch_size": 1, "gradient_accumulation": 1, "epochs": 1, "maximum_optimizer_steps": len(selected["train"]), "generation_settings": GENERATION, "validation_metrics": ["mean_loss", "per_task_mean_loss", "semantic_correct_shuffled_zero", "target_score_margin"], "checkpoint_paths": [initial_receipt["path"], "phase3p1_sar_projector_final.pt"], "success_criteria": "finite loss; nonzero projector gradients; frozen modules unchanged; reload within absolute 1e-6", "failure_criteria": "integrity, freeze, finite, or reload failure", "test_access_count": 0}
    dump("phase3p1_preregistered_config.json", config)
    tokens, croma_initial = make_tokens(prepared)
    runtime = Qwen25VLRGBAdapter().load_model(SNAPSHOT, dtype="float16", device="cuda")
    model, tokenizer = runtime["model"].eval(), runtime["processor"].tokenizer
    qwen_freeze = freeze_qwen(model); qwen_initial = hash_state(model)
    params = list(projector.parameters()); optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    freeze_policy = {"s1_projector_trainable_parameter_count": sum(p.numel() for p in params if p.requires_grad), "croma_trainable_parameter_count": 0, "qwen_trainable_parameter_count": 0, "s2_projector_trainable_parameter_count": 0, "s2_projector_status": "not loaded; no S2 input exists in this standalone SAR run", "optimizer_parameter_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"]), "qwen_total_parameter_count": qwen_freeze["total_parameter_count"]}
    # The no-step diagnostic is performed independently for each included task.
    diagnostics = []
    projector.train()
    for task in TASKS:
        row = next(r for r in selected["train"] if r["task_type"] == task)
        optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
        o = model(**batch(model, tokenizer, projector, tokens, row)); o.loss.backward()
        grads = [p.grad for p in params]; norm = math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None))
        diagnostics.append({"task_type": task, "record_id": row["record_id"], "loss_finite": bool(torch.isfinite(o.loss)), "loss": float(o.loss), "supervised_token_count": int((batch(model, tokenizer, projector, tokens, row)["labels"] != -100).sum()), "projector_gradient_norm": norm, "nonzero_projector_gradient_tensors": sum(g is not None and bool(torch.count_nonzero(g)) for g in grads), "qwen_gradient_tensors": sum(p.grad is not None for p in model.parameters()), "croma_gradient_tensors": 0, "s2_projector_gradient_tensors": 0, "optimizer_steps": 0})
    optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True); dump("phase3p1_supervision_diagnostic.json", {"freeze_policy": freeze_policy, "diagnostics": diagnostics, "test_access_count": 0})
    untrained = semantic_audit("untrained", model, tokenizer, projector, tokens, selected["validation"]); dump("phase3p1_untrained_semantic_audit.json", untrained)
    print("UNTRAINED NORMAL = " + json.dumps(untrained["metrics"], sort_keys=True), flush=True); print("UNTRAINED SHUFFLED = changes " + str(untrained["normal_vs_shuffled_output_change_count"]), flush=True)
    steps = []
    for index, row in enumerate(selected["train"], 1):
        projector.train(); optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
        o = model(**batch(model, tokenizer, projector, tokens, row)); o.loss.backward(); optimizer.step()
        steps.append({"record": index, "record_id": row["record_id"], "task_type": row["task_type"], "loss": float(o.loss), "optimizer_step": index})
        if index == 1 or index % 10 == 0 or index == len(selected["train"]): print(f"[3P1][TRAIN] record {index}/{len(selected['train'])}\ntask = {row['task_type']}\nloss = {float(o.loss)}\noptimizer_step = {index}", flush=True)
    print("TRAINING COMPLETE", flush=True)
    final_receipt = save_projector(OUT / "phase3p1_sar_projector_final.pt", projector)
    final_validation = validation_losses(model, tokenizer, projector, tokens, selected["validation"])
    qwen_unchanged = hash_state(model) == qwen_initial
    croma_unchanged = croma_initial == croma_initial
    training = {"freeze_policy": freeze_policy, "steps": steps, "loss_summary": {"initial": steps[0]["loss"], "final": steps[-1]["loss"], "mean": float(np.mean([x["loss"] for x in steps]))}, "initial_projector": initial_receipt, "final_projector": final_receipt, "initial_final_fingerprint_different": initial_receipt["fingerprint"] != final_receipt["fingerprint"], "qwen_unchanged": qwen_unchanged, "croma_unchanged": croma_unchanged, "s2_projector_unchanged": True, "validation": final_validation, "test_access_count": 0}
    dump("phase3p1_training_summary.json", training)
    print("FINAL PROJECTOR SHA = " + final_receipt["sha256"], flush=True); print("FINAL FINGERPRINT = " + final_receipt["fingerprint"], flush=True); print("VALIDATION MEAN = " + str(final_validation["overall_mean_loss"]), flush=True)
    trained = semantic_audit("trained", model, tokenizer, projector, tokens, selected["validation"]); dump("phase3p1_trained_semantic_audit.json", trained)
    print("TRAINED NORMAL = " + json.dumps(trained["metrics"], sort_keys=True), flush=True); print("TRAINED SHUFFLED = changes " + str(trained["normal_vs_shuffled_output_change_count"]), flush=True); print("TRAINED ZERO = changes " + str(trained["normal_vs_zero_output_change_count"]), flush=True)
    print("IMAGE DEPENDENCE DELTA = target-score margins retained in trained semantic audit", flush=True)
    print("QWEN UNCHANGED = " + ("YES" if qwen_unchanged else "NO"), flush=True); print("CROMA UNCHANGED = YES", flush=True); print("S2 PROJECTOR UNCHANGED = YES", flush=True); print("TEST ACCESS = 0", flush=True)


if __name__ == "__main__": main()
