"""Fresh-process, inference-only reproducibility receipt for Phase 3Q.1."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.croma_adapter import CROMAAdapter
from src.dataset_loader import discover_samples, load_sample
from src.eo_vlm.joint_croma_projector import CromaJointProjector
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.sar_projector import S1SARProjector
from src.eo_vlm.training import build_qwen_visual_token_batch, freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT, hash_state

OUT = ROOT / "artifacts/training/phase3q/phase3q1_fusion_adaptation"
MANIFEST = OUT / "phase3q1_training_manifest.json"
SHUFFLE = OUT / "phase3q1_shuffle_map.json"
SUMMARY = OUT / "phase3q1_training_summary.json"
FINAL = OUT / "phase3q1_joint_projector_final.pt"
AUDIT = OUT / "phase3q1_trained_semantic_audit.json"
PHASE_SUMMARY = OUT / "phase3q1_summary.json"
RECEIPT = OUT / "phase3q1_reload_receipt.json"
DATA = Path(r"D:\Satquery_ai datasets\comparison\raw-1000")
CROMA_SOURCE = Path(r"D:\Satquery_ai datasets\croma_official")
CROMA_CHECKPOINT = Path(r"D:\Satquery_ai datasets\checkpoints\CROMA_base.pt")
S1_CHECKPOINT = ROOT / "artifacts/training/phase3p/phase3p1_sar_adaptation/phase3p1_sar_projector_final.pt"
S2_CHECKPOINT = ROOT / "artifacts/training/phase3o/phase3o12_run/phase3o12_projector_final.pt"
TASKS = ("binary_qa", "multiple_choice_qa", "caption")
EXPECTED_SHA = "bedcb8dc60d3375878b6fe732cb90b568283c7d5565148fb1a52eae0d1b12eaa"
EXPECTED_FINGERPRINT = "8b3e05aa2346c24f662d747ab59d47fee8c2a96dc1728e87d2dce6b30a4d1a69"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def question(record: dict) -> str:
    return "Describe this paired Sentinel-1 and Sentinel-2 scene." if record["task_type"] == "caption" else record["question"]


def answer(record: dict) -> str:
    return record["caption"] if record["task_type"] == "caption" else record["answer"]


def require_parseable(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON object: {path.name}")
    return value


def main() -> None:
    print("PHASE3Q1 RELOAD START", flush=True)
    manifest_bytes_before = sha256(MANIFEST)
    shuffle_bytes_before = sha256(SHUFFLE)
    training_summary = require_parseable(SUMMARY)
    trained_audit = require_parseable(AUDIT)
    phase_summary = require_parseable(PHASE_SUMMARY)
    manifest = require_parseable(MANIFEST)
    shuffle = require_parseable(SHUFFLE)
    rows = manifest["selection"]["validation"]
    task_counts = Counter(row["task_type"] for row in rows)
    if len(rows) != 30 or dict(task_counts) != {task: 10 for task in TASKS}:
        raise RuntimeError("PHASE3Q1_FAILED_INTEGRITY: validation manifest is not exact 30 (10 per task)")
    if manifest["test_access_count"] != 0 or shuffle["test_access_count"] != 0:
        raise RuntimeError("PHASE3Q1_FAILED_INTEGRITY: test access is nonzero")
    if len(trained_audit.get("records", [])) != 30 or trained_audit.get("test_access_count") != 0:
        raise RuntimeError("PHASE3Q1_FAILED_INTEGRITY: trained audit is invalid")

    final_sha = sha256(FINAL)
    projector = CromaJointProjector().eval()
    projector.load_state_dict(torch.load(FINAL, map_location="cpu", weights_only=True)["state_dict"])
    final_fingerprint = fingerprint(projector)
    print(f"FINAL SHA = {final_sha}", flush=True)
    print(f"FINAL FINGERPRINT = {final_fingerprint}", flush=True)
    if final_sha != EXPECTED_SHA or final_fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError("PHASE3Q1_FAILED_INTEGRITY: final checkpoint identity mismatch")
    if final_sha != training_summary["final"]["sha256"] or final_fingerprint != training_summary["final"]["fingerprint"]:
        raise RuntimeError("PHASE3Q1_FAILED_INTEGRITY: saved summary identity mismatch")

    # CROMA is loaded fresh, frozen, and used only to regenerate correct paired tokens.
    samples = {sample.patch_id: sample for sample in discover_samples(DATA, strict=True)}
    if any(row["image_id"] not in samples for row in rows):
        raise RuntimeError("PHASE3Q1_FAILED_INTEGRITY: validation sample missing")
    croma = CROMAAdapter(CROMA_SOURCE, CROMA_CHECKPOINT, device="cuda")
    for parameter in croma.model.parameters():
        parameter.requires_grad_(False)
    croma_before = fingerprint(croma.model)
    token_cache: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for row in rows:
            prepared = load_sample(samples[row["image_id"]])
            token_cache[row["record_id"]] = croma.infer(prepared.raw_optical, prepared.raw_sar)["joint_encodings"].cpu()
    croma_unchanged = fingerprint(croma.model) == croma_before
    del croma
    gc.collect()
    torch.cuda.empty_cache()

    # These historical projectors remain on CPU and are fingerprint-checked only.
    s1 = S1SARProjector().eval()
    s1.load_state_dict(torch.load(S1_CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"])
    for parameter in s1.parameters():
        parameter.requires_grad_(False)
    s1_before = fingerprint(s1)
    s2 = S2MultispectralProjector().eval()
    s2.load_state_dict(torch.load(S2_CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"])
    for parameter in s2.parameters():
        parameter.requires_grad_(False)
    s2_before = fingerprint(s2)

    runtime = Qwen25VLRGBAdapter().load_model(SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"].eval()
    tokenizer = runtime["processor"].tokenizer
    freeze_qwen(model)
    qwen_before = hash_state(model)
    projector = projector.cuda().eval()
    projector_before = fingerprint(projector)
    losses: list[dict] = []
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            visual_tokens = projector(token_cache[row["record_id"]].cuda())
            batch = build_qwen_visual_token_batch(
                tokenizer=tokenizer,
                model=model,
                visual_tokens=visual_tokens,
                questions=[question(row)],
                answers=[answer(row)],
            )
            loss = model(**batch).loss
            losses.append({"record_id": row["record_id"], "task_type": row["task_type"], "loss": float(loss)})
            if index in {1, 5, 10, 15, 20, 25, 30}:
                print(f"[3Q1][RELOAD] record {index}/30", flush=True)

    all_finite = all(math.isfinite(item["loss"]) for item in losses)
    reload_mean = float(np.mean([item["loss"] for item in losses]))
    per_task = {task: float(np.mean([item["loss"] for item in losses if item["task_type"] == task])) for task in TASKS}
    original_mean = float(training_summary["validation"]["mean"])
    absolute_difference = abs(reload_mean - original_mean)
    agreement = "EXACT_MATCH" if absolute_difference == 0.0 else "WITHIN_TOLERANCE" if absolute_difference <= 1e-6 else "MISMATCH"

    qwen_unchanged = hash_state(model) == qwen_before
    joint_unchanged = fingerprint(projector) == projector_before
    s1_unchanged = fingerprint(s1) == s1_before
    s2_unchanged = fingerprint(s2) == s2_before
    manifest_unchanged = sha256(MANIFEST) == manifest_bytes_before
    shuffle_unchanged = sha256(SHUFFLE) == shuffle_bytes_before
    immutable = all((qwen_unchanged, croma_unchanged, s1_unchanged, s2_unchanged, joint_unchanged, manifest_unchanged, shuffle_unchanged))
    status = "PHASE3Q1_COMPLETE" if agreement != "MISMATCH" and all_finite and immutable else "PHASE3Q1_FAILED_REPRODUCIBILITY"
    classification = phase_summary.get("classification_pending_reload")
    receipt = {
        "status": status,
        "final_checkpoint_sha256": final_sha,
        "final_projector_fingerprint": final_fingerprint,
        "qwen_revision": str(SNAPSHOT),
        "validation_count": len(rows),
        "task_counts": dict(task_counts),
        "original_precise_validation_mean": original_mean,
        "reload_precise_validation_mean": reload_mean,
        "absolute_difference": absolute_difference,
        "per_task_reload_means": per_task,
        "all_losses_finite": all_finite,
        "agreement": agreement,
        "immutability": {
            "qwen": qwen_unchanged,
            "croma": croma_unchanged,
            "s1_projector": s1_unchanged,
            "s2_projector": s2_unchanged,
            "joint_projector": joint_unchanged,
            "manifest": manifest_unchanged,
            "shuffle_map": shuffle_unchanged,
            "no_optimizer_created": True,
        },
        "test_access_count": 0,
        "existing_trained_audit_valid": True,
        "existing_fusion_classification": classification,
        "raw_reload_losses": losses,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ORIGINAL VALIDATION MEAN = {original_mean}", flush=True)
    print(f"RELOAD VALIDATION MEAN = {reload_mean}", flush=True)
    print(f"ABSOLUTE DIFFERENCE = {absolute_difference}", flush=True)
    print(f"BINARY RELOAD MEAN = {per_task['binary_qa']}", flush=True)
    print(f"MCQ RELOAD MEAN = {per_task['multiple_choice_qa']}", flush=True)
    print(f"CAPTION RELOAD MEAN = {per_task['caption']}", flush=True)
    print(f"ALL LOSSES FINITE = {'YES' if all_finite else 'NO'}", flush=True)
    print(f"RELOAD AGREEMENT = {agreement}", flush=True)
    print(f"QWEN UNCHANGED = {'YES' if qwen_unchanged else 'NO'}", flush=True)
    print(f"CROMA UNCHANGED = {'YES' if croma_unchanged else 'NO'}", flush=True)
    print(f"S1 PROJECTOR UNCHANGED = {'YES' if s1_unchanged else 'NO'}", flush=True)
    print(f"S2 PROJECTOR UNCHANGED = {'YES' if s2_unchanged else 'NO'}", flush=True)
    print(f"JOINT PROJECTOR UNCHANGED DURING RELOAD = {'YES' if joint_unchanged else 'NO'}", flush=True)
    print("TEST ACCESS = 0", flush=True)
    print("EXISTING TRAINED AUDIT VALID = YES", flush=True)
    print(f"FINAL FUSION CLASSIFICATION = {classification}", flush=True)
    print(f"PHASE3Q1 STATUS = {status}", flush=True)
    print(f"RELOAD RECEIPT PATH = {RECEIPT}", flush=True)


if __name__ == "__main__":
    main()
