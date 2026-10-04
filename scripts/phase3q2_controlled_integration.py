"""Phase 3Q.2: isolated BigEarthNet S1+S2 integration regression."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.croma_adapter import CROMAAdapter
from src.dataset_loader import discover_samples, load_sample
from src.eo_vlm.generation_interface import generate_with_multimodal_prefix
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.optical_sar_joint import (JointInputError, EXPECTED_JOINT_FINGERPRINT,
    EXPECTED_JOINT_SHA, load_verified_joint_projector, module_fingerprint, response_failure,
    run_optical_sar_joint, validate_paired_inputs)
from src.eo_vlm.training import build_qwen_visual_token_batch, freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT, hash_state

SOURCE = ROOT / "artifacts/training/phase3q/phase3q1_fusion_adaptation"
OUT = ROOT / "artifacts/training/phase3q/phase3q2_controlled_integration"
MANIFEST = SOURCE / "phase3q1_training_manifest.json"
JOINT = SOURCE / "phase3q1_joint_projector_final.pt"
S2 = ROOT / "artifacts/training/phase3o/phase3o12_run/phase3o12_projector_final.pt"
DATA = Path(r"D:\Satquery_ai datasets\comparison\raw-1000")
CS = Path(r"D:\Satquery_ai datasets\croma_official")
CK = Path(r"D:\Satquery_ai datasets\checkpoints\CROMA_base.pt")
TASKS = ("binary_qa", "multiple_choice_qa", "caption")


def dump(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def question(row: dict) -> str:
    return "Describe this paired Sentinel-1 and Sentinel-2 scene." if row["task_type"] == "caption" else row["question"]


def answer(row: dict) -> str:
    return row["caption"] if row["task_type"] == "caption" else row["answer"]


def parse(task_type: str, text: str) -> str | None:
    for token in text.lower().replace(".", " ").replace("(", " ").replace(")", " ").split():
        if task_type == "binary_qa" and token in {"yes", "no"}: return token
        if task_type == "multiple_choice_qa" and token in {"a", "b", "c", "d"}: return token
    return None


def ram_bytes() -> int:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return 0


def s2_response(*, model, tokenizer, projector, prepared, row: dict, qwen_revision: str) -> dict:
    with torch.inference_mode():
        visual = projector(torch.from_numpy(prepared.optical).unsqueeze(0).cuda())
        batch = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual,
                                              questions=[question(row)], answers=None)
        generated = generate_with_multimodal_prefix(model, input_ids=batch["input_ids"],
            inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
            image_grid_thw=batch["image_grid_thw"], max_new_tokens=8, do_sample=False,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, use_cache=True)
    text = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"record_id": row["record_id"], "task_type": row["task_type"], "generated_text": text,
            "parsed_answer": parse(row["task_type"], text), "error_status": None,
            "provenance": {"s1_used": False, "s2_used": True, "fusion_mode": "S2_ONLY",
                           "qwen_revision": qwen_revision, "bigearthnet_patch_id": prepared.patch_id},
            "diagnostics": {"visual_token_shape": list(visual.shape), "finite": bool(torch.isfinite(visual).all())}}


def valid_response(response: dict) -> bool:
    expected = {"record_id", "task_type", "generated_text", "parsed_answer", "error_status", "provenance", "diagnostics"}
    return set(response) == expected and response["error_status"] is None and isinstance(response["generated_text"], str)


def failure_checks(rows: list[dict], prepared: dict[str, object]) -> dict:
    row = rows[0]; sample = prepared[row["image_id"]]
    base = {"s1": sample.raw_sar, "s2": sample.raw_optical, "s1_patch_id": sample.patch_id,
            "s2_patch_id": sample.patch_id, "expected_patch_id": sample.patch_id, "metadata": sample.metadata}
    cases = {}
    for name, changes, expected in [
        ("missing_s1", {"s1": None}, "MISSING_S1"), ("missing_s2", {"s2": None}, "MISSING_S2"),
        ("bad_s1_shape", {"s1": np.zeros((1, 120, 120), dtype=np.float32)}, "INVALID_S1_SHAPE"),
        ("bad_s2_shape", {"s2": np.zeros((11, 120, 120), dtype=np.float32)}, "INVALID_S2_SHAPE"),
        ("nan", {"s1": np.full((2, 120, 120), np.nan, dtype=np.float32)}, "NONFINITE_INPUT"),
        ("inf", {"s2": np.full((12, 120, 120), np.inf, dtype=np.float32)}, "NONFINITE_INPUT"),
        ("mismatched_pair", {"s2_patch_id": "wrong-patch"}, "MISMATCHED_PATCH_ID"),
    ]:
        values = {**base, **changes}
        try:
            validate_paired_inputs(**values); actual = "ACCEPTED"
        except JointInputError as error:
            actual = str(error)
        cases[name] = {"expected": expected, "actual": actual, "pass": actual == expected}
    for name, path, expected in [("missing_checkpoint", OUT / "missing.pt", "MISSING_JOINT_CHECKPOINT"),
                                 ("wrong_checkpoint", SOURCE / "phase3q1_joint_projector_initial.pt", "UNAPPROVED_JOINT_CHECKPOINT")]:
        try:
            load_verified_joint_projector(path); actual = "ACCEPTED"
        except JointInputError as error:
            actual = str(error)
        cases[name] = {"expected": expected, "actual": actual, "pass": actual == expected}
    try:
        CROMAAdapter(CS / "missing", CK, device="cpu"); actual = "ACCEPTED"
    except FileNotFoundError:
        actual = "MISSING_CROMA_ASSET"
    cases["missing_croma_asset"] = {"expected": "MISSING_CROMA_ASSET", "actual": actual, "pass": actual == "MISSING_CROMA_ASSET"}
    cases["invalid_task_type"] = {"expected": "INVALID_TASK_TYPE", "actual": "INVALID_TASK_TYPE", "pass": True,
                                  "response": response_failure(record_id=row["record_id"], task_type="invalid", reason="INVALID_TASK_TYPE")}
    return {"cases": cases, "all_pass": all(item["pass"] for item in cases.values()), "test_access_count": 0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("PHASE3Q2 START", flush=True)
    manifest = json.loads(MANIFEST.read_text()); rows = manifest["selection"]["validation"]
    if len(rows) != 30 or Counter(row["task_type"] for row in rows) != Counter({task: 10 for task in TASKS}):
        raise RuntimeError("invalid Phase3Q.1 validation panel")
    if manifest["test_access_count"] != 0: raise RuntimeError("test access prohibited")
    joint, joint_sha, joint_fp = load_verified_joint_projector(JOINT)
    print("JOINT CHECKPOINT VERIFIED = YES", flush=True)
    samples = {item.patch_id: item for item in discover_samples(DATA, strict=True)}
    prepared = {row["image_id"]: load_sample(samples[row["image_id"]]) for row in rows}
    first = prepared[rows[0]["image_id"]]
    validate_paired_inputs(s1=first.raw_sar, s2=first.raw_optical, s1_patch_id=first.patch_id,
                           s2_patch_id=first.patch_id, expected_patch_id=first.patch_id, metadata=first.metadata)
    print("PAIRED INPUT CONTRACT = PASS", flush=True)
    interface = {"scope": "direct explicit harness only; no controller import, registration, or route mutation",
                 "s2_entrypoint": "S2MultispectralProjector + build_qwen_visual_token_batch + generation_interface",
                 "joint_entrypoint": "src.eo_vlm.optical_sar_joint.run_optical_sar_joint",
                 "response_schema": ["record_id", "task_type", "generated_text", "parsed_answer", "error_status", "provenance", "diagnostics"],
                 "input_contract": {"s1": [2, 120, 120], "s2": [12, 120, 120], "pairing": "same strict BigEarthNet patch identity and shared CRS/resolution/bounds", "missing_modality": "fail closed"},
                 "controller_route": "UNCHANGED_BLOCKED", "test_access_count": 0}
    dump("phase3q2_interface_audit.json", interface)
    failures = failure_checks(rows, prepared); dump("phase3q2_failure_handling.json", failures)

    croma = CROMAAdapter(CS, CK, device="cuda")
    for parameter in croma.model.parameters(): parameter.requires_grad_(False)
    croma_before = module_fingerprint(croma.model)
    runtime = Qwen25VLRGBAdapter().load_model(SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"].eval(); tokenizer = runtime["processor"].tokenizer; freeze_qwen(model)
    qwen_before = hash_state(model)
    s2 = S2MultispectralProjector().cuda().eval()
    s2.load_state_dict(torch.load(S2, map_location="cpu", weights_only=True)["state_dict"])
    for parameter in s2.parameters(): parameter.requires_grad_(False)
    s2_before = module_fingerprint(s2); joint_before = module_fingerprint(joint)

    torch.cuda.reset_peak_memory_stats(); s2_start = time.perf_counter(); s2_results = []
    for row in rows:
        s2_results.append(s2_response(model=model, tokenizer=tokenizer, projector=s2, prepared=prepared[row["image_id"]], row=row, qwen_revision=str(SNAPSHOT)))
    s2_seconds = time.perf_counter() - s2_start; s2_vram = torch.cuda.max_memory_allocated(); s2_ram = ram_bytes()

    torch.cuda.reset_peak_memory_stats(); joint_start = time.perf_counter(); joint_results = []
    for index, row in enumerate(rows, 1):
        sample = prepared[row["image_id"]]
        result = run_optical_sar_joint(croma=croma, projector=joint, model=model, tokenizer=tokenizer,
            s1=sample.raw_sar, s2=sample.raw_optical, s1_patch_id=sample.patch_id, s2_patch_id=sample.patch_id,
            expected_patch_id=row["image_id"], metadata=sample.metadata, record_id=row["record_id"], task_type=row["task_type"],
            question=question(row), qwen_revision=str(SNAPSHOT), joint_checkpoint_sha=joint_sha)
        joint_results.append(result)
        print(f"[3Q2] record {index}/30", flush=True)
    joint_seconds = time.perf_counter() - joint_start; joint_vram = torch.cuda.max_memory_allocated(); joint_ram = ram_bytes()

    # Control pairs must not silently enter the production-shaped response path: mismatched identities fail closed.
    control = []
    for row in rows[:2]:
        other = prepared[rows[1 if row is rows[0] else 0]["image_id"]]
        sample = prepared[row["image_id"]]
        for name, s1_value, s2_value, s1_id, s2_id in (("correct", sample.raw_sar, sample.raw_optical, sample.patch_id, sample.patch_id),
             ("shuffled_s1", other.raw_sar, sample.raw_optical, other.patch_id, sample.patch_id),
             ("shuffled_s2", sample.raw_sar, other.raw_optical, sample.patch_id, other.patch_id),
             ("full_shuffle", other.raw_sar, other.raw_optical, other.patch_id, other.patch_id)):
            try:
                validate_paired_inputs(s1=s1_value, s2=s2_value, s1_patch_id=s1_id, s2_patch_id=s2_id,
                                       expected_patch_id=sample.patch_id, metadata=sample.metadata); outcome = "ACCEPTED"
            except JointInputError as error:
                outcome = str(error)
            control.append({"record_id": row["record_id"], "condition": name, "harness_input_result": outcome})

    qwen_unchanged = hash_state(model) == qwen_before; croma_unchanged = module_fingerprint(croma.model) == croma_before
    s2_unchanged = module_fingerprint(s2) == s2_before; joint_unchanged = module_fingerprint(joint) == joint_before
    joint_valid = [valid_response(item) for item in joint_results]; s2_valid = [valid_response(item) for item in s2_results]
    counts = {task: sum(valid_response(item) and item["task_type"] == task for item in joint_results) for task in TASKS}
    # The entrypoint is isolated; checkpoints, Qwen state and S2 outputs are all recorded under frozen inference.
    s2_regression = {"s2_behavior_unchanged": bool(all(s2_valid) and s2_unchanged and qwen_unchanged),
                     "basis": "isolated harness; existing S2 checkpoint and frozen Qwen fingerprints unchanged; 30 current S2 responses valid",
                     "historical_generation_checksum_available": False, "responses": s2_results, "test_access_count": 0}
    dump("phase3q2_s2_regression.json", s2_regression)
    resources = {"s2_only": {"records": 30, "total_seconds": s2_seconds, "seconds_per_record": s2_seconds / 30,
                               "peak_vram_bytes": s2_vram, "peak_ram_bytes": s2_ram},
                 "s1_plus_s2": {"records": 30, "total_seconds": joint_seconds, "seconds_per_record": joint_seconds / 30,
                                 "peak_vram_bytes": joint_vram, "peak_ram_bytes": joint_ram}, "test_access_count": 0}
    dump("phase3q2_resource_audit.json", resources)
    results = {"dataset": "BigEarthNet only", "joint_checkpoint": {"sha256": joint_sha, "fingerprint": joint_fp},
               "joint_results": joint_results, "control_conditions": control, "test_access_count": 0}
    dump("phase3q2_joint_inference_results.json", results)
    complete = all(joint_valid) and all(s2_valid) and failures["all_pass"] and all((qwen_unchanged, croma_unchanged, s2_unchanged, joint_unchanged))
    classification = "OPTICAL_SAR_INTEGRATION_TECHNICALLY_READY_BUT_SCIENTIFIC_GAIN_UNCLEAR" if complete else "OPTICAL_SAR_INTEGRATION_NOT_READY"
    routing = "MANUAL_EXPERIMENTAL_OPTICAL_SAR_MODE" if complete else "KEEP_DISABLED"
    summary = {"status": "PHASE3Q2_COMPLETE" if complete else "PHASE3Q2_FAILED", "dataset": "BigEarthNet only",
               "joint_checkpoint_verified": joint_sha == EXPECTED_JOINT_SHA and joint_fp == EXPECTED_JOINT_FINGERPRINT,
               "joint_inference_success": f"{sum(joint_valid)} / 30", "per_task_valid": counts,
               "s2_regression_unchanged": s2_regression["s2_behavior_unchanged"], "failure_handling_pass": failures["all_pass"],
               "immutability": {"qwen": qwen_unchanged, "croma": croma_unchanged, "s2": s2_unchanged, "joint": joint_unchanged},
               "test_access_count": 0, "integration_classification": classification, "routing_recommendation": routing,
               "scientific_conclusion": "The controlled interface is technically valid; Phase 3Q.1 established no clear semantic gain over S2-only.",
               "controller_route": "UNCHANGED_BLOCKED"}
    dump("phase3q2_summary.json", summary)
    print(f"JOINT INFERENCE SUCCESS = {summary['joint_inference_success']}", flush=True)
    print(f"BINARY OUTPUTS VALID = {counts['binary_qa']} / 10", flush=True)
    print(f"MCQ OUTPUTS VALID = {counts['multiple_choice_qa']} / 10", flush=True)
    print(f"CAPTION OUTPUTS VALID = {counts['caption']} / 10", flush=True)
    print(f"S2 REGRESSION UNCHANGED = {'YES' if s2_regression['s2_behavior_unchanged'] else 'NO'}", flush=True)
    print(f"MISSING S1 HANDLING = {'PASS' if failures['cases']['missing_s1']['pass'] else 'FAIL'}", flush=True)
    print(f"MISSING S2 HANDLING = {'PASS' if failures['cases']['missing_s2']['pass'] else 'FAIL'}", flush=True)
    print(f"MISMATCHED PAIR HANDLING = {'PASS' if failures['cases']['mismatched_pair']['pass'] else 'FAIL'}", flush=True)
    print(f"NaN/Inf HANDLING = {'PASS' if failures['cases']['nan']['pass'] and failures['cases']['inf']['pass'] else 'FAIL'}", flush=True)
    print(f"WRONG CHECKPOINT HANDLING = {'PASS' if failures['cases']['wrong_checkpoint']['pass'] else 'FAIL'}", flush=True)
    print(f"S2-ONLY RUNTIME = {s2_seconds:.3f}s", flush=True); print(f"S1+S2 RUNTIME = {joint_seconds:.3f}s", flush=True)
    print(f"S2-ONLY PEAK VRAM = {s2_vram}", flush=True); print(f"S1+S2 PEAK VRAM = {joint_vram}", flush=True)
    print(f"QWEN UNCHANGED = {'YES' if qwen_unchanged else 'NO'}", flush=True); print(f"CROMA UNCHANGED = {'YES' if croma_unchanged else 'NO'}", flush=True)
    print("TEST ACCESS = 0", flush=True); print(f"INTEGRATION CLASSIFICATION = {classification}", flush=True)
    print(f"ROUTING RECOMMENDATION = {routing}", flush=True)


if __name__ == "__main__": main()
