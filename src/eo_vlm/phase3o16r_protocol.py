"""Fail-closed gates for the Phase 3O.16R recovery experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED = {
    "binary": {"margin_increased", "normal_accuracy_not_decreased", "normal_beats_shuffled_by_one", "parseability_not_decreased", "output_changed", "losses_finite", "projector_gradients_nonzero", "qwen_gradients_zero", "test_access_zero"},
    "mcq": {"margin_increased", "image_margin_not_decreased", "normal_accuracy_not_decreased", "normal_beats_shuffled_by_one", "parseability_not_worse", "losses_finite", "projector_gradients_nonzero", "qwen_gradients_zero", "test_access_zero"},
    "caption": {"margin_increased", "unique_not_decreased", "dominant_not_increased", "empty_malformed_not_increased", "output_changed", "prefix_diversity_not_materially_worse", "losses_finite", "projector_gradients_nonzero", "qwen_gradients_zero", "test_access_zero"},
}

def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def validate_preregistration(value: dict) -> None:
    gates = value.get("success_gates", {})
    for task, fields in REQUIRED.items():
        actual = gates.get(task, {}).get("all_required", [])
        if set(actual) != fields:
            raise ValueError(f"incomplete {task} success gate")
    if value.get("preregistration_sha256") != canonical_sha({k:v for k,v in value.items() if k != "preregistration_sha256"}):
        raise ValueError("preregistration digest mismatch")

def classify(task: str, evidence: dict) -> tuple[str, list[str]]:
    missing = REQUIRED[task] - set(evidence)
    if missing:
        return "NOT_PROMISING", ["missing evidence: " + ", ".join(sorted(missing))]
    failures = sorted(key for key in REQUIRED[task] if evidence[key] is not True)
    return ("PROMISING", []) if not failures else ("NOT_PROMISING", failures)

def frozen_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
