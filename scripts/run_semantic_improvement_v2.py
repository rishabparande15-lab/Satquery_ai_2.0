"""Bounded v2 SAR and joint-projector experiments, TRAIN/DEV only for selection.

The original validation VQA rows are historically exposed, so they are used
once only after DEV selection and are explicitly not acceptance-quality.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
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
from src.config import get_settings
from src.croma_adapter import CROMAAdapter
from src.dataset_loader import OPTICAL_BANDS, SAR_BANDS, Sample, load_sample
from src.eo_vlm.image_conditioned_objective import image_conditioned_loss, target_sequence_log_probability
from src.eo_vlm.joint_croma_projector import CromaJointProjector
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch, freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from src.satquery_v1 import JOINT
from src.single_image_sar_vqa import PROJECTOR_CHECKPOINT as SAR_CHECKPOINT
from src.single_image_vqa import QWEN_SNAPSHOT
from scripts.run_semantic_improvement import close as close_controller, constrained as constrained_answer, make_controller

OUT = ROOT / "artifacts" / "semantic_improvement_v2"
SPLITS = OUT / "splits.json"
SEED = 20261004
ROUTES = ("S2", "SAR", "JOINT")


def dump(name: str, value: object) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(message, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(row: dict) -> list[str]:
    return ["yes", "no"] if row["task_type"] == "binary_qa" else [item["key"] for item in row["options"]]


def metric(rows: list[dict]) -> dict:
    result = {}
    for task in ("binary_qa", "multiple_choice_qa"):
        group = [row for row in rows if row["task_type"] == task]
        correct = sum(row["prediction"] == row["answer"] for row in group)
        result[task] = {"correct": correct, "total": len(group), "accuracy": correct / len(group) if group else None}
    result["aggregate"] = {"correct": sum(row["prediction"] == row["answer"] for row in rows), "total": len(rows), "accuracy": sum(row["prediction"] == row["answer"] for row in rows) / len(rows) if rows else None}
    return result


def load_samples(rows: list[dict]) -> dict[str, object]:
    settings = get_settings()
    ids = {row["image_id"] for row in rows} | {row["shuffled_image_id"] for row in rows}
    table = parquet.read_table(settings.dataset_root / "metadata.parquet", columns=["patch_id", "s1_name", "split"], filters=[("patch_id", "in", sorted(ids))])
    values = {str(item["patch_id"]): item for item in table.to_pylist()}
    if set(values) != ids or any(str(item["split"]) not in {"train", "validation"} for item in values.values()):
        raise RuntimeError("NON_DEVELOPMENT_METADATA_RETURNED")
    samples = {}
    for patch_id in sorted(ids):
        item = values[patch_id]; s1_id = str(item["s1_name"])
        s2_scene, s1_scene = patch_id.rsplit("_", 2)[0], s1_id.rsplit("_", 3)[0]
        optical = {band: settings.dataset_root / "BigEarthNet-S2" / s2_scene / patch_id / f"{patch_id}_{band}.tif" for band in OPTICAL_BANDS}
        sar = {band: settings.dataset_root / "BigEarthNet-S1" / s1_scene / s1_id / f"{s1_id}_{band}.tif" for band in SAR_BANDS}
        reference = settings.dataset_root / "Reference_Maps" / s2_scene / patch_id / f"{patch_id}_reference_map.tif"
        samples[patch_id] = load_sample(Sample(patch_id, optical, sar, reference, str(item["split"])))
    return samples


def make_shuffle(rows: list[dict]) -> list[dict]:
    output = []
    for task in ("binary_qa", "multiple_choice_qa"):
        group = sorted((dict(row) for row in rows if row["task_type"] == task), key=lambda row: row["record_id"])
        images = [row["image_id"] for row in group]
        for row, shuffled in zip(group, images[1:] + images[:1]):
            row["shuffled_image_id"] = shuffled
            output.append(row)
    return output


def load_runtime():
    runtime = Qwen25VLRGBAdapter().load_model(QWEN_SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"].eval(); tokenizer = runtime["processor"].tokenizer
    if freeze_qwen(model)["qwen_trainable_parameter_count"] != 0:
        raise RuntimeError("QWEN_FREEZE_FAILED")
    return model, tokenizer


def cache_features(kind: str, samples: dict[str, object]) -> dict[str, torch.Tensor]:
    settings = get_settings(); croma = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device="cuda")
    for parameter in croma.model.parameters(): parameter.requires_grad_(False)
    output = {}
    try:
        for index, (image_id, sample) in enumerate(samples.items(), 1):
            with torch.no_grad():
                output[image_id] = (croma.infer_modality(sar=sample.sar.astype(np.float32, copy=False))["SAR_encodings"] if kind == "SAR" else croma.infer(sample.optical, sample.sar)["joint_encodings"]).detach().cpu()
            if index == 1 or index % 8 == 0 or index == len(samples): log(f"[CACHE {kind}] {index}/{len(samples)}")
    finally:
        del croma; gc.collect(); torch.cuda.empty_cache()
    return output


def projector(kind: str) -> torch.nn.Module:
    module = S1SARProjector() if kind == "SAR" else CromaJointProjector()
    checkpoint = SAR_CHECKPOINT if kind == "SAR" else JOINT
    module.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    return module.cuda()


def score(model, tokenizer, module, features, row: dict, image_id: str, *, require_grad: bool = False):
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        visual = module(features[image_id].to("cuda"))
        scores = {}
        for candidate in candidates(row):
            batch = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual, questions=[row["question"]], answers=[candidate])
            scores[candidate] = target_sequence_log_probability(model(**batch).logits, batch["labels"])
    return scores


def evaluate(kind: str, module, model, tokenizer, features, rows: list[dict], label: str) -> tuple[list[dict], dict]:
    module.eval(); records = []
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            correct_scores = score(model, tokenizer, module, features, row, row["image_id"])
            shuffled_scores = score(model, tokenizer, module, features, row, row["shuffled_image_id"])
            prediction = max(correct_scores, key=lambda key: float(correct_scores[key]))
            shuffled_prediction = max(shuffled_scores, key=lambda key: float(shuffled_scores[key]))
            records.append({"record_id": row["record_id"], "task_type": row["task_type"], "answer": row["answer"], "prediction": prediction, "shuffled_prediction": shuffled_prediction, "correct": prediction == row["answer"], "decision_changed": prediction != shuffled_prediction, "score_margin_correct_minus_shuffled": float(correct_scores[row["answer"]] - shuffled_scores[row["answer"]])})
            log(f"[EVAL {label} {index}/{len(rows)}] route={kind} task={row['task_type']} correct={prediction == row['answer']} changed={prediction != shuffled_prediction}")
    return records, {**metric(records), "shuffle_decision_change": sum(row["decision_changed"] for row in records) / len(records), "mean_target_margin": float(np.mean([row["score_margin_correct_minus_shuffled"] for row in records]))}


def production_baseline(rows: list[dict], samples: dict[str, object], label: str) -> tuple[list[dict], dict]:
    records = []
    for route in ROUTES:
        controller = make_controller(route); controller.load()
        try:
            for index, row in enumerate(rows, 1):
                result = constrained_answer(controller, route, row, samples[row["image_id"]])
                records.append({"route": route, "record_id": row["record_id"], "task_type": row["task_type"], "answer": row["answer"], "prediction": result["prediction"], "correct": result["prediction"] == row["answer"]})
                log(f"[BASELINE {label} {index}/{len(rows)}] route={route} task={row['task_type']} correct={result['prediction'] == row['answer']}")
        finally:
            close_controller(controller)
    return records, {route: metric([record for record in records if record["route"] == route]) for route in ROUTES}


def train_experiment(kind: str, train: list[dict], dev: list[dict], features: dict[str, torch.Tensor], name: str):
    exp = OUT / "experiments" / name; exp.mkdir(parents=True, exist_ok=True)
    prereg = {"hypothesis": f"Frozen Qwen with a {kind} projector can increase correct-image target likelihood over deterministic shuffled-image conditioning.", "architecture": f"frozen Qwen + frozen official CROMA + trainable current {kind} projector", "seed": SEED, "train_record_ids": [row["record_id"] for row in train], "dev_record_ids": [row["record_id"] for row in dev], "optimizer": "AdamW", "learning_rate": 0.0001, "batch_size": 1, "maximum_optimizer_steps": len(train) * 2, "loss_terms": "Qwen answer cross-entropy + 0.25*hinge(0.20 - correct_target_logprob + shuffled_target_logprob)", "checkpoint_selection_rule": "highest DEV aggregate accuracy, then highest DEV shuffle decision-change rate", "stopping_rule": "two fixed epochs only", "test_access_count": 0}
    (exp / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model, tokenizer = load_runtime(); module = projector(kind); baseline, base_metrics = evaluate(kind, module, model, tokenizer, features, dev, f"{name}:baseline")
    opt = torch.optim.AdamW(module.parameters(), lr=1e-4); steps = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(2):
        module.train()
        for idx, row in enumerate(train, 1):
            opt.zero_grad(set_to_none=True)
            visual = module(features[row["image_id"]].to("cuda")); shuffled_visual = module(features[row["shuffled_image_id"]].to("cuda"))
            good = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual, questions=[row["question"]], answers=[row["answer"]])
            bad = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=shuffled_visual, questions=[row["question"]], answers=[row["answer"]])
            good_out = model(**good); bad_out = model(**bad)
            losses = image_conditioned_loss(good_out.logits, bad_out.logits, good["labels"], base_lm_loss=good_out.loss, contrastive_weight=0.25, margin=0.20)
            losses["total_loss"].backward()
            norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)); opt.step()
            step = epoch * len(train) + idx
            steps.append({"step": step, "answer_loss": float(losses["base_lm_loss"]), "shuffle_loss": float(losses["contrastive_loss"]), "total_loss": float(losses["total_loss"]), "score_margin": float(losses["correct_score"] - losses["shuffled_score"]), "grad_norm": norm})
            if step == 1 or step % 6 == 0 or step == len(train) * 2:
                log(f"[TRAIN {step}/{len(train)*2}] route={kind} loss={steps[-1]['total_loss']:.4f} answer_loss={steps[-1]['answer_loss']:.4f} shuffle_loss={steps[-1]['shuffle_loss']:.4f} grad_norm={norm:.4f} gpu_allocated={torch.cuda.memory_allocated()} gpu_reserved={torch.cuda.memory_reserved()} elapsed={time.perf_counter()-started:.1f}")
    candidate, candidate_metrics = evaluate(kind, module, model, tokenizer, features, dev, f"{name}:candidate")
    checkpoint = exp / f"{kind.lower()}_projector.pt"; torch.save({"state_dict": module.state_dict()}, checkpoint)
    result = {"route": kind, "baseline_dev_metrics": base_metrics, "candidate_dev_metrics": candidate_metrics, "baseline_predictions": baseline, "candidate_predictions": candidate, "trainable_parameters": sum(parameter.numel() for parameter in module.parameters()), "qwen_frozen": True, "optimizer_steps": len(steps), "final_train_loss": steps[-1]["total_loss"], "training_steps": steps, "checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint), "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(), "test_access_count": 0}
    (exp / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del module, model; gc.collect(); torch.cuda.empty_cache()
    return result, checkpoint


def main() -> None:
    if not SPLITS.is_file(): raise RuntimeError("RUN_PREPARE_FIRST")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    split = json.loads(SPLITS.read_text(encoding="utf-8")); train, dev = make_shuffle(split["train"]), make_shuffle(split["dev"])
    locked = make_shuffle(split["historically_exposed_locked_validation"])
    samples = load_samples(train + dev)
    baseline_dev_records, baseline_dev_metrics = production_baseline(dev, samples, "dev")
    dump("baseline_dev_predictions.json", baseline_dev_records); dump("baseline_dev_metrics.json", baseline_dev_metrics)
    features_sar = cache_features("SAR", samples); features_joint = cache_features("JOINT", samples)
    sar_result, sar_checkpoint = train_experiment("SAR", train, dev, features_sar, "experiment_01_sar_shuffle_ranking")
    joint_result, joint_checkpoint = train_experiment("JOINT", train, dev, features_joint, "experiment_02_joint_shuffle_ranking")
    selected = "JOINT" if joint_result["candidate_dev_metrics"]["aggregate"]["accuracy"] >= sar_result["candidate_dev_metrics"]["aggregate"]["accuracy"] else "SAR"
    best_result, best_checkpoint = (joint_result, joint_checkpoint) if selected == "JOINT" else (sar_result, sar_checkpoint)
    dump("dev_comparison.json", {"SAR": sar_result["candidate_dev_metrics"], "JOINT": joint_result["candidate_dev_metrics"], "selected": selected, "selection_scope": "DEV only", "test_access_count": 0})
    dump("best_candidate_config.json", {"selected": selected, "selection_rule": "DEV aggregate only", "production_deployment": "forbidden", "test_access_count": 0})
    dump("best_candidate_checkpoint_hashes.json", {"route": selected, "sha256": sha(best_checkpoint), "path": str(best_checkpoint.relative_to(ROOT)), "test_access_count": 0})
    # Final historically-exposed comparison is performed once; it cannot be used
    # to claim clean generalization because v1 already evaluated these identities.
    locked_samples = load_samples(locked); locked_features = cache_features(selected, locked_samples)
    model, tokenizer = load_runtime(); module = projector(selected); module.load_state_dict(torch.load(best_checkpoint, map_location="cpu", weights_only=True)["state_dict"]); module.cuda()
    baseline_module = projector(selected)
    baseline_records, baseline_metrics = evaluate(selected, baseline_module, model, tokenizer, locked_features, locked, "locked-baseline")
    candidate_records, candidate_metrics = evaluate(selected, module, model, tokenizer, locked_features, locked, "locked-candidate")
    dump("baseline_locked_predictions.json", baseline_records); dump("baseline_locked_metrics.json", baseline_metrics)
    dump("locked_validation_predictions.json", candidate_records); dump("locked_validation_metrics.json", candidate_metrics)
    dump("modality_dependence.json", {"selected": selected, "baseline": {"shuffle_decision_change": baseline_metrics["shuffle_decision_change"], "mean_target_margin": baseline_metrics["mean_target_margin"]}, "candidate": {"shuffle_decision_change": candidate_metrics["shuffle_decision_change"], "mean_target_margin": candidate_metrics["mean_target_margin"]}, "status": "historically exposed validation; diagnostic only", "test_access_count": 0})
    dump("final_summary.json", {"selected": selected, "sar_dev": sar_result["candidate_dev_metrics"], "joint_dev": joint_result["candidate_dev_metrics"], "locked_baseline": baseline_metrics, "locked_candidate": candidate_metrics, "fresh_locked_validation_available": False, "test_access_count": 0})
    log("SATQUERY_SEMANTIC_IMPROVEMENT_V2_RUN_COMPLETE")


if __name__ == "__main__": main()
