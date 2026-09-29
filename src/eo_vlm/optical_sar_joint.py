"""Controlled, non-routed S1+S2 CROMA-to-Qwen inference harness.

This module deliberately has no controller dependency.  Callers must opt in
directly and provide a validated BigEarthNet paired sample.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .generation_interface import generate_with_multimodal_prefix
from .joint_croma_projector import CromaJointProjector
from .training import build_qwen_visual_token_batch

EXPECTED_JOINT_SHA = "bedcb8dc60d3375878b6fe732cb90b568283c7d5565148fb1a52eae0d1b12eaa"
EXPECTED_JOINT_FINGERPRINT = "8b3e05aa2346c24f662d747ab59d47fee8c2a96dc1728e87d2dce6b30a4d1a69"
TASK_TYPES = frozenset({"binary_qa", "multiple_choice_qa", "caption"})


class JointInputError(ValueError):
    """A fail-closed paired-input or task-contract error."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_paired_inputs(*, s1: np.ndarray | None, s2: np.ndarray | None,
                           s1_patch_id: str | None, s2_patch_id: str | None,
                           expected_patch_id: str, metadata: Mapping[str, Any] | None) -> None:
    """Validate exact paired BigEarthNet tensors and shared spatial metadata."""
    if s1 is None:
        raise JointInputError("MISSING_S1")
    if s2 is None:
        raise JointInputError("MISSING_S2")
    if not isinstance(s1, np.ndarray) or s1.shape != (2, 120, 120):
        raise JointInputError("INVALID_S1_SHAPE")
    if not isinstance(s2, np.ndarray) or s2.shape != (12, 120, 120):
        raise JointInputError("INVALID_S2_SHAPE")
    if not np.isfinite(s1).all() or not np.isfinite(s2).all():
        raise JointInputError("NONFINITE_INPUT")
    if not expected_patch_id or s1_patch_id != expected_patch_id or s2_patch_id != expected_patch_id:
        raise JointInputError("MISMATCHED_PATCH_ID")
    if not metadata or not metadata.get("crs") or not metadata.get("resolution") or not metadata.get("bounds"):
        raise JointInputError("INCOMPLETE_SPATIAL_PROVENANCE")
    resolution = metadata["resolution"]
    bounds = metadata["bounds"]
    if len(resolution) != 2 or len(bounds) != 4 or not np.isfinite(np.asarray(resolution + bounds, dtype=float)).all():
        raise JointInputError("INVALID_SPATIAL_PROVENANCE")
    if any(float(value) <= 0 for value in resolution) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise JointInputError("INVALID_SPATIAL_PROVENANCE")


def load_verified_joint_projector(path: Path, *, device: str = "cuda") -> tuple[CromaJointProjector, str, str]:
    """Load only the approved Phase 3Q.1 checkpoint, fail closed otherwise."""
    path = Path(path)
    if not path.is_file():
        raise JointInputError("MISSING_JOINT_CHECKPOINT")
    checkpoint_sha = sha256_file(path)
    if checkpoint_sha != EXPECTED_JOINT_SHA:
        raise JointInputError("UNAPPROVED_JOINT_CHECKPOINT")
    projector = CromaJointProjector().eval()
    projector.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    projector_fingerprint = module_fingerprint(projector)
    if projector_fingerprint != EXPECTED_JOINT_FINGERPRINT:
        raise JointInputError("UNAPPROVED_JOINT_CHECKPOINT")
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    return projector.to(device).eval(), checkpoint_sha, projector_fingerprint


def parse_answer(task_type: str, generated_text: str) -> str | None:
    for token in generated_text.lower().replace(".", " ").replace("(", " ").replace(")", " ").split():
        if task_type == "binary_qa" and token in {"yes", "no"}:
            return token
        if task_type == "multiple_choice_qa" and token in {"a", "b", "c", "d"}:
            return token
    return None


def response_failure(*, record_id: str | None, task_type: str | None, reason: str) -> dict[str, Any]:
    return {"record_id": record_id, "task_type": task_type, "generated_text": None,
            "parsed_answer": None, "error_status": reason, "provenance": None, "diagnostics": {}}


def run_optical_sar_joint(*, croma: Any, projector: CromaJointProjector, model: Any, tokenizer: Any,
                          s1: np.ndarray | None, s2: np.ndarray | None, s1_patch_id: str | None,
                          s2_patch_id: str | None, expected_patch_id: str, metadata: Mapping[str, Any] | None, record_id: str,
                          task_type: str, question: str, qwen_revision: str, joint_checkpoint_sha: str,
                          max_new_tokens: int = 8) -> dict[str, Any]:
    """Run an explicitly requested joint branch and return the common response contract."""
    try:
        if task_type not in TASK_TYPES:
            raise JointInputError("INVALID_TASK_TYPE")
        validate_paired_inputs(s1=s1, s2=s2, s1_patch_id=s1_patch_id, s2_patch_id=s2_patch_id,
                               expected_patch_id=expected_patch_id, metadata=metadata)
        with torch.inference_mode():
            joint_tokens = croma.infer(s2, s1)["joint_encodings"]
            if not torch.isfinite(joint_tokens).all():
                raise JointInputError("NONFINITE_JOINT_TOKENS")
            visual_tokens = projector(joint_tokens)
            if tuple(visual_tokens.shape[1:]) != (16, 2048) or not torch.isfinite(visual_tokens).all():
                raise JointInputError("INVALID_VISUAL_TOKENS")
            batch = build_qwen_visual_token_batch(tokenizer=tokenizer, model=model, visual_tokens=visual_tokens,
                                                   questions=[question], answers=None)
            generated = generate_with_multimodal_prefix(
                model, input_ids=batch["input_ids"], inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"], image_grid_thw=batch["image_grid_thw"],
                max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0, use_cache=True,
            )
        text = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"record_id": record_id, "task_type": task_type, "generated_text": text,
                "parsed_answer": parse_answer(task_type, text), "error_status": None,
                "provenance": {"s1_used": True, "s2_used": True, "fusion_mode": "CROMA_JOINT",
                               "joint_projector_sha256": joint_checkpoint_sha, "qwen_revision": qwen_revision,
                               "bigearthnet_patch_id": s1_patch_id},
                "diagnostics": {"joint_token_shape": list(joint_tokens.shape),
                                "visual_token_shape": list(visual_tokens.shape), "finite": True}}
    except FileNotFoundError:
        return response_failure(record_id=record_id, task_type=task_type, reason="MISSING_CROMA_ASSET")
    except (JointInputError, ValueError) as error:
        return response_failure(record_id=record_id, task_type=task_type, reason=str(error))
