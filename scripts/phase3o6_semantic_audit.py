"""Phase 3O.6 held-out semantic generation audit; inference only, never trains."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase3o2_adaptation_pilot import (  # noqa: E402
    MODEL_REVISION, MODEL_SNAPSHOT, cube, hash_state,
)
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file  # noqa: E402
from src.eo_vlm.multispectral_projector import (  # noqa: E402
    QWEN_IMAGE_GRID_THW, QWEN_IMAGE_TOKEN_COUNT, S2MultispectralProjector,
)
from src.eo_vlm.training import freeze_qwen  # noqa: E402
from src.eo_vlm_adapter import Qwen25VLRGBAdapter  # noqa: E402

OUT = ROOT / "artifacts/training/phase3o"
RUN = OUT / "phase3o5_run"
CHECKPOINT = RUN / "s2_multispectral_projector.pt"
AUDIT_PATH = OUT / "phase3o6_semantic_audit.json"
EXPECTED_SHA256 = "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db"
EXPECTED_FP = "db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
GENERATION_CONFIG = {
    "max_new_tokens": 16,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": 1.0,
    "use_cache": True,
    "prompt_template": "Question: {question}\\nAnswer:",
    "tokenizer_configuration": {"add_special_tokens": True, "chat_template": "not_used"},
}


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def historical_s2_hash(paths: dict[str, Path]) -> str:
    payload = [(band, sha256_file(paths[band])) for band in S2_BANDS]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def build_generation_inputs(tokenizer, model, projector, s2: torch.Tensor, question: str):
    """Build the exact learned-S2 visual prefix used in Phase 3O.5, without labels."""
    device = next(model.parameters()).device
    visual = projector(s2.to(device))
    prompt = GENERATION_CONFIG["prompt_template"].format(question=question)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prefix = [model.config.vision_start_token_id]
    prefix += [model.config.image_token_id] * QWEN_IMAGE_TOKEN_COUNT
    prefix += [model.config.vision_end_token_id]
    input_ids = torch.tensor([prefix + prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    embeddings = model.get_input_embeddings()(input_ids)
    image_mask = input_ids.eq(model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
    visual_flat = visual.to(dtype=embeddings.dtype).reshape(-1)
    if int(image_mask.sum()) != visual.numel():
        raise RuntimeError("GENERATION_BLOCKED_VISUAL_TOKEN_MISMATCH")
    embeddings = embeddings.masked_scatter(image_mask, visual_flat)
    return input_ids, attention_mask, embeddings


def normalize_binary(text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*(yes|no|true|false)(?=$|[\s.,:;!?])", text, flags=re.I)
    if not match:
        return None, "unparsable_binary"
    token = match.group(1).lower()
    return ({"true": "yes", "false": "no"}.get(token, token)), "parsed_binary"


def normalize_mcq(text: str, options: list[dict] | None) -> tuple[str | None, str]:
    match = re.match(r"^\s*(?:option\s+)?(?:\()?(?P<key>[a-z])(?:\))?(?=$|[\s.,:;!?])", text, flags=re.I)
    allowed = {str(option["key"]).lower() for option in (options or [])}
    if match and match.group("key").lower() in allowed:
        return match.group("key").lower(), "parsed_option_key"
    normalized = re.sub(r"\s+", " ", text.strip().lower()).rstrip(" .!?")
    exact_matches = [str(option["key"]).lower() for option in (options or [])
                     if normalized == re.sub(r"\s+", " ", str(option["text"]).strip().lower()).rstrip(" .!?")]
    if len(exact_matches) == 1:
        return exact_matches[0], "parsed_exact_option_text"
    return None, "unparsable_mcq"


def degenerate_flags(text: str, prompt: str) -> list[str]:
    stripped = text.strip()
    flags: list[str] = []
    if not stripped:
        flags.append("empty")
    if stripped and not any(char.isalnum() for char in stripped):
        flags.append("punctuation_or_special_only")
    if prompt.lower() in text.lower():
        flags.append("prompt_echo")
    tokens = re.findall(r"\w+", stripped.lower())
    if len(tokens) >= 4 and len(set(tokens)) == 1:
        flags.append("excessive_repetition")
    return flags


def main() -> None:
    started = time.perf_counter()
    qwen_snapshot_verified = MODEL_SNAPSHOT.is_dir() and MODEL_SNAPSHOT.name == MODEL_REVISION
    if not qwen_snapshot_verified:
        raise RuntimeError("PHASE3O6_BLOCKED_QWEN_REVISION_MISMATCH")
    if not CHECKPOINT.is_file():
        raise RuntimeError("PHASE3O6_BLOCKED_CHECKPOINT_MISSING")
    checkpoint_sha = sha256_file(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    projector_state = checkpoint["state_dict"]
    projector_fp = state_fingerprint(projector_state)
    if checkpoint_sha != EXPECTED_SHA256 or projector_fp != EXPECTED_FP:
        raise RuntimeError("PHASE3O6_BLOCKED_CHECKPOINT_IDENTITY_MISMATCH")

    pilot = json.loads((RUN / "pilot_manifest.json").read_text(encoding="utf-8"))
    validation = pilot["selection"]["validation"]
    train_ids = {record["image_id"] for record in pilot["selection"]["train"]}
    validation_ids = {record["image_id"] for record in validation}
    if len(validation) != 20 or len(validation_ids) != 20 or train_ids & validation_ids:
        raise RuntimeError("PHASE3O6_BLOCKED_VALIDATION_MANIFEST_MISMATCH")
    annotations = {json.loads(line)["record_id"]: json.loads(line)
                   for line in (ROOT / "artifacts/annotations/image_language/visual_vqa_candidates.jsonl").open(encoding="utf-8")}
    assets: list[tuple[dict, dict, dict[str, Path]]] = []
    for manifest_record in validation:
        row = annotations.get(manifest_record["record_id"])
        if row is None or row["split"] != "validation" or row["task_type"] not in {"binary_qa", "multiple_choice_qa", "caption"}:
            raise RuntimeError("PHASE3O6_BLOCKED_RECORD_PROVENANCE_MISMATCH")
        root = Path(manifest_record["source_root"])
        paths = {band: next(root.rglob(f"{row['image_id']}_{band}.tif"), None) for band in S2_BANDS}
        if any(path is None for path in paths.values()):
            raise RuntimeError("PHASE3O6_BLOCKED_S2_ASSET_MISSING")
        if historical_s2_hash(paths) != manifest_record["s2_source_hash"]:
            raise RuntimeError("PHASE3O6_BLOCKED_S2_PROVENANCE_MISMATCH")
        asset = inspect_area(row, paths).to_dict()
        assets.append((manifest_record, row, asset))

    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    adapter = Qwen25VLRGBAdapter()
    runtime = adapter.load_model(MODEL_SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"]
    model.eval()
    freeze_summary = freeze_qwen(model)
    if freeze_summary["qwen_trainable_parameter_count"] != 0:
        raise RuntimeError("PHASE3O6_BLOCKED_QWEN_NOT_FROZEN")
    projector = S2MultispectralProjector().to("cuda")
    projector.load_state_dict(projector_state)
    projector.eval()
    qwen_before = hash_state(model)
    projector_before = hash_state(projector)
    rows: list[dict] = []

    with torch.inference_mode():
        for manifest_record, row, asset in assets:
            prompt = GENERATION_CONFIG["prompt_template"].format(question=row["question"])
            input_ids, attention_mask, inputs_embeds = build_generation_inputs(
                runtime["processor"].tokenizer, model, projector, cube(asset), row["question"],
            )
            output_ids = model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], dtype=torch.long, device=input_ids.device),
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
                use_cache=True,
            )
            generated_ids = output_ids[0, input_ids.shape[1]:]
            generated_text = runtime["processor"].tokenizer.decode(generated_ids, skip_special_tokens=True)
            task_type = row["task_type"]
            if task_type == "binary_qa":
                prediction, parse_status = normalize_binary(generated_text)
            elif task_type == "multiple_choice_qa":
                prediction, parse_status = normalize_mcq(generated_text, row.get("options"))
            else:
                prediction, parse_status = (generated_text.strip() or None), "caption_descriptive_only"
            correct = prediction == row["answer"] if prediction is not None and task_type != "caption" else None
            rows.append({
                "sample_id": row["record_id"], "image_id": row["image_id"], "split": "validation",
                "task_type": task_type, "question": row["question"], "prompt": prompt,
                "allowed_answer_choices": row.get("options"), "reference": row["answer"],
                "generated_text": generated_text, "normalized_prediction": prediction,
                "correct": correct, "parse_status": parse_status,
                "generation_token_count": int(generated_ids.numel()), "degenerate_flags": degenerate_flags(generated_text, prompt),
                "checkpoint_sha256": checkpoint_sha, "model_state_fingerprint": projector_fp,
                "qwen_revision": MODEL_REVISION, "generation_config": GENERATION_CONFIG,
                "provenance": {"annotation_id": row["annotation_id"], "s2_source_hash": manifest_record["s2_source_hash"],
                               "source_root": manifest_record["source_root"], "annotation_source_revision": row["source_revision"]},
            })

    qwen_after = hash_state(model)
    projector_after = hash_state(projector)
    checkpoint_sha_after = sha256_file(CHECKPOINT)
    projector_state_after = state_fingerprint(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"])
    if checkpoint_sha_after != checkpoint_sha or projector_state_after != projector_fp:
        raise RuntimeError("PHASE3O6_FAILED_CHECKPOINT_MUTATED")
    if qwen_before != qwen_after or projector_before != projector_after:
        raise RuntimeError("PHASE3O6_FAILED_INFERENCE_MUTATED_PARAMETERS")

    task_counts = Counter(row["task_type"] for row in rows)
    summaries = {}
    for task in ("binary_qa", "multiple_choice_qa"):
        subset = [row for row in rows if row["task_type"] == task]
        correct = sum(row["correct"] is True for row in subset)
        unparsable = sum(row["normalized_prediction"] is None for row in subset)
        summaries[task] = {"total": len(subset), "correct": correct, "incorrect": len(subset) - correct - unparsable,
                           "unparsable": unparsable, "accuracy": (correct / len(subset)) if subset else None}
    all_texts = [row["generated_text"].strip() for row in rows]
    nonempty = [text for text in all_texts if text]
    repeated_all = len(set(nonempty)) == 1 if nonempty else False
    receipt = {
        "phase": "3O.6", "status": "PHASE3O6_COMPLETE", "audit_scope": "internal_controlled_validation_generation_only",
        "validation_count": len(rows), "test_access_count": 0, "test_records_accessed": [],
        "train_validation_overlap": len(train_ids & validation_ids), "task_distribution": dict(task_counts),
        "task_results": summaries, "caption_audit": {"count": task_counts["caption"], "metrics": "not_applicable_no_caption_records"},
        "unparsable_generation_count": sum(row["normalized_prediction"] is None for row in rows),
        "degenerate_generation": {"empty": sum(not text for text in all_texts), "identical_nonempty_response_to_every_example": repeated_all,
                                  "flag_counts": dict(Counter(flag for row in rows for flag in row["degenerate_flags"]))},
        "generation_config": GENERATION_CONFIG, "samples": rows,
        "checkpoint_sha256": checkpoint_sha, "expected_checkpoint_sha256": EXPECTED_SHA256, "checkpoint_sha256_verified": checkpoint_sha == EXPECTED_SHA256,
        "model_state_fingerprint": projector_fp, "expected_model_state_fingerprint": EXPECTED_FP, "model_state_fingerprint_verified": projector_fp == EXPECTED_FP,
        "qwen_revision": MODEL_REVISION, "qwen_snapshot_path": str(MODEL_SNAPSHOT),
        "qwen_revision_verified": qwen_snapshot_verified, "qwen_frozen": True, "qwen_trainable_parameter_count": 0,
        "qwen_immutability_verified": qwen_before == qwen_after, "projector_immutability_verified": projector_before == projector_after,
        "checkpoint_unchanged_after_inference": checkpoint_sha_after == checkpoint_sha,
        "no_optimizer_created": True, "zero_gradient_updates": True, "provenance_status": "VERIFIED",
        "untrained_baseline_status": "UNTRAINED_BASELINE_UNAVAILABLE", "benchmark_inference_executed": False,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "peak_ram_bytes": process.memory_info().rss,
        "duration_seconds": time.perf_counter() - started,
    }
    AUDIT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
