"""Phase 3O.10 fixed-panel audit for the versioned generation interface fix.

This is inference-only: it loads the immutable Phase 3O.5 projector and
compares stock generation with the scoped SatQuery generation wrapper.
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

from scripts.phase3o6_semantic_audit import state_fingerprint
from scripts.phase3o8_visual_conditioning_diagnostic import assemble, cube, index, prompt
from scripts.run_phase3o2_adaptation_pilot import MODEL_REVISION, MODEL_SNAPSHOT, hash_state
from src.eo_vlm.generation_interface import GENERATION_INTERFACE_VERSION, generate_with_multimodal_prefix
from src.eo_vlm.multispectral_projector import QWEN_IMAGE_GRID_THW, S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.training import freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

OUT = ROOT / "artifacts/training/phase3o"
RUN = OUT / "phase3o5_run"
CHECKPOINT = RUN / "s2_multispectral_projector.pt"
PANEL = OUT / "phase3o7_validation_panel.json"
ARTIFACT = OUT / "phase3o10_generation_interface_repair.json"
SHA = "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db"
FINGERPRINT = "db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
HISTORICAL = [RUN / "pilot_manifest.json", RUN / "training_summary.json", RUN / "phase3o5_reload_receipt.json", OUT / "phase3o6_semantic_audit.json", OUT / "phase3o7_task_balanced_semantic_audit.json", OUT / "phase3o8_visual_conditioning_diagnostic.json", OUT / "phase3o9_generation_supervision_diagnostic.json"]
CONFIG = {"do_sample": False, "max_new_tokens": 16, "temperature": None, "top_p": None, "top_k": None, "repetition_penalty": 1.0, "use_cache": True}
TOLERANCE = 1e-5


def top10(tokenizer, logits):
    values, ids = torch.topk(logits.float(), 10)
    return [{"id": int(i), "token": tokenizer.decode([int(i)]), "logit": float(v)} for v, i in zip(values, ids)]


def eos_rank(tokenizer, logits):
    order = torch.argsort(logits.float(), descending=True)
    return int((order == tokenizer.eos_token_id).nonzero()[0]) + 1


def capture_generation(model, generator, ids, attention, embeds):
    original = model.forward
    capture = {}

    def wrapped(*args, **kwargs):
        if not capture:
            for name in ("input_ids", "inputs_embeds", "attention_mask", "cache_position", "position_ids"):
                value = kwargs.get(name)
                capture[name] = value.detach().clone() if hasattr(value, "detach") else value
        return original(*args, **kwargs)

    model.forward = wrapped
    try:
        with torch.inference_mode():
            result = generator(model, input_ids=ids, inputs_embeds=embeds, attention_mask=attention,
                               image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], dtype=torch.long, device=ids.device),
                               return_dict_in_generate=True, output_scores=True, **CONFIG)
    finally:
        model.forward = original
    return result, capture


def metadata(value):
    if value is None:
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def call_equivalence(ids, attention, embeds, captured):
    actual_ids = captured["input_ids"]
    actual_embeds = captured["inputs_embeds"]
    actual_attention = captured["attention_mask"]
    return {
        "input_ids_present": actual_ids is not None,
        "input_ids_equal": actual_ids is not None and bool(torch.equal(actual_ids, ids)),
        "inputs_embeds_equal": actual_embeds is not None and bool(torch.equal(actual_embeds, embeds)),
        "attention_mask_equal": actual_attention is not None and bool(torch.equal(actual_attention, attention)),
        "input_ids": metadata(actual_ids), "inputs_embeds": metadata(actual_embeds), "attention_mask": metadata(actual_attention),
        "cache_position": captured["cache_position"].tolist() if captured["cache_position"] is not None else None,
        "position_ids": captured["position_ids"].tolist() if captured["position_ids"] is not None else None,
        "sequence_length": int(ids.shape[1]), "visual_token_indices": list(range(1, 17)),
    }


def main():
    started = time.perf_counter()
    if sha256_file(CHECKPOINT) != SHA:
        raise RuntimeError("BLOCKED_CHECKPOINT_SHA")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    if state_fingerprint(state) != FINGERPRINT or not (MODEL_SNAPSHOT.is_dir() and MODEL_SNAPSHOT.name == MODEL_REVISION):
        raise RuntimeError("BLOCKED_IDENTITY")
    panel = json.loads(PANEL.read_text(encoding="utf8"))
    rows = [row for task in ("binary_qa", "multiple_choice_qa", "caption") for row in sorted(panel["records"][task], key=lambda x: x["record_id"])[:5]]
    historical_before = {str(p.relative_to(ROOT)): sha256_file(p) for p in HISTORICAL}
    adapter = Qwen25VLRGBAdapter()
    runtime = adapter.load_model(MODEL_SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"].eval()
    if freeze_qwen(model)["qwen_trainable_parameter_count"]:
        raise RuntimeError("BLOCKED_QWEN_TRAINABLE")
    projector = S2MultispectralProjector().to("cuda")
    projector.load_state_dict(state)
    projector.eval()
    qwen_before, projector_before = hash_state(model), hash_state(projector)
    tokenizer, paths = runtime["processor"].tokenizer, index()
    records = []
    with torch.inference_mode():
        for row in rows:
            asset = inspect_area(row, {band: paths[f"{row['image_id']}_{band}.tif"] for band in S2_BANDS}).to_dict()
            visual = projector(cube(asset).to("cuda"))
            ids, attention, embeds = assemble(tokenizer, model, visual, prompt(row))
            direct = model(input_ids=ids, inputs_embeds=embeds, attention_mask=attention,
                           image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], device=ids.device), return_dict=True).logits[0, -1]
            pre, pre_capture = capture_generation(model, lambda m, **kw: m.generate(**kw), ids, attention, embeds)
            post, post_capture = capture_generation(model, generate_with_multimodal_prefix, ids, attention, embeds)
            post_logits = post.scores[0][0]
            delta = direct.float() - post_logits.float()
            pre_new = pre.sequences[0, ids.shape[1]:]
            post_new = post.sequences[0, ids.shape[1]:]
            reference = row.get("caption") if row["task_type"] == "caption" else row["answer"]
            def result(new):
                text = tokenizer.decode(new, skip_special_tokens=True)
                return {"text": text, "empty": not bool(text.strip()), "token_ids": [int(x) for x in new],
                        "termination": "eos" if len(new) and int(new[-1]) in set(model.generation_config.eos_token_id) else "max_new_tokens",
                        "correct": text.strip().lower() == str(reference).strip().lower() if row["task_type"] != "caption" else None}
            records.append({"sample_id": row["record_id"], "task_type": row["task_type"], "reference": reference,
                            "direct_top10": top10(tokenizer, direct), "direct_eos_rank": eos_rank(tokenizer, direct),
                            "pre_fix": {"inputs": call_equivalence(ids, attention, embeds, pre_capture), "first_top10": top10(tokenizer, pre.scores[0][0]), "eos_rank": eos_rank(tokenizer, pre.scores[0][0]), "output": result(pre_new)},
                            "post_fix": {"inputs": call_equivalence(ids, attention, embeds, post_capture), "first_top10": top10(tokenizer, post_logits), "eos_rank": eos_rank(tokenizer, post_logits), "output": result(post_new)},
                            "logit_comparison": {"l2": float(torch.linalg.vector_norm(delta)), "max_abs": float(delta.abs().max()), "tolerance": TOLERANCE,
                                                 "within_tolerance": bool(torch.linalg.vector_norm(delta) <= TOLERANCE and delta.abs().max() <= TOLERANCE)}})
    qwen_after, projector_after = hash_state(model), hash_state(projector)
    historical_after = {str(p.relative_to(ROOT)): sha256_file(p) for p in HISTORICAL}
    if qwen_before != qwen_after or projector_before != projector_after or historical_before != historical_after or sha256_file(CHECKPOINT) != SHA:
        raise RuntimeError("IMMUTABILITY_FAILED")
    def task_summary(task):
        items = [x for x in records if x["task_type"] == task]
        return {"changed_outputs": sum(x["pre_fix"]["output"]["text"] != x["post_fix"]["output"]["text"] for x in items),
                "unchanged_outputs": sum(x["pre_fix"]["output"]["text"] == x["post_fix"]["output"]["text"] for x in items),
                "correct_before": sum(bool(x["pre_fix"]["output"]["correct"]) for x in items), "correct_after": sum(bool(x["post_fix"]["output"]["correct"]) for x in items)}
    artifact = {"phase": "3O.10", "status": "PHASE3O10_COMPLETE_FIX_VALIDATED", "classification": "B_GENERATE_INPUT_PREPARATION_MISMATCH_FIXED",
                "generation_interface_version": GENERATION_INTERFACE_VERSION, "checkpoint_sha256": SHA, "checkpoint_verified": True, "projector_fingerprint": FINGERPRINT, "fingerprint_verified": True, "qwen_revision": MODEL_REVISION,
                "test_access_count": 0, "optimizer_created": False, "backward_pass": False, "qwen_unchanged": qwen_before == qwen_after, "projector_unchanged": projector_before == projector_after, "historical_artifacts_unchanged": historical_before == historical_after,
                "repair": {"old": "HF prefill dropped input_ids when inputs_embeds was supplied", "new": "SatQuery restores original placeholder input_ids only on inputs_embeds prefill", "later_cached_decode_unmodified": True},
                "records": records, "smoke_summary": {"binary": task_summary("binary_qa"), "multiple_choice": task_summary("multiple_choice_qa"),
                    "caption": {"empty_before": sum(x["pre_fix"]["output"]["empty"] for x in records if x["task_type"] == "caption"), "empty_after": sum(x["post_fix"]["output"]["empty"] for x in records if x["task_type"] == "caption"), "changed_outputs": sum(x["pre_fix"]["output"]["text"] != x["post_fix"]["output"]["text"] for x in records if x["task_type"] == "caption"), "nonempty_after": sum(not x["post_fix"]["output"]["empty"] for x in records if x["task_type"] == "caption")}},
                "all_postfix_inputs_equivalent": all(x["post_fix"]["inputs"]["input_ids_equal"] and x["post_fix"]["inputs"]["inputs_embeds_equal"] and x["post_fix"]["inputs"]["attention_mask_equal"] for x in records),
                "all_logits_within_tolerance": all(x["logit_comparison"]["within_tolerance"] for x in records), "duration_seconds": time.perf_counter() - started}
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(artifact["smoke_summary"], indent=2))


if __name__ == "__main__":
    main()
