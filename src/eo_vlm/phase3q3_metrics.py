"""Deterministic aggregation helpers for the fixed Phase 3Q.3 comparison."""
from __future__ import annotations
from collections import Counter
from typing import Iterable

CONDITIONS = ("S2_ONLY_CORRECT", "S1_PLUS_S2_CORRECT", "SHUFFLED_S1_CORRECT_S2",
              "CORRECT_S1_SHUFFLED_S2", "FULLY_SHUFFLED_S1_S2", "ZERO_S1_CORRECT_S2", "CORRECT_S1_ZERO_S2")


def outcome(row: dict) -> str:
    if row.get("parsed_answer") is None: return "unparsable"
    return "correct" if row.get("parsed_answer") == row.get("target") else "incorrect"


def paired_wins(s2: Iterable[dict], joint: Iterable[dict]) -> dict:
    wins = losses = ties = 0
    for left, right in zip(s2, joint, strict=True):
        left_ok, right_ok = outcome(left) == "correct", outcome(right) == "correct"
        if right_ok and not left_ok: wins += 1
        elif left_ok and not right_ok: losses += 1
        else: ties += 1
    return {"joint_wins": wins, "joint_losses": losses, "ties": ties}


def qa_summary(rows: Iterable[dict]) -> dict:
    values = list(rows); counts = Counter(outcome(row) for row in values)
    return {"count": len(values), "accuracy": counts["correct"] / len(values) if values else 0.0,
            "parseability": (len(values) - counts["unparsable"]) / len(values) if values else 0.0,
            "outcomes": dict(counts)}


def changed(correct: Iterable[dict], altered: Iterable[dict]) -> int:
    return sum(a.get("generated_text") != b.get("generated_text") for a, b in zip(correct, altered, strict=True))


def task_classification(s2: dict, joint: dict) -> str:
    if joint["accuracy"] > s2["accuracy"]: return "JOINT_BETTER"
    if joint["accuracy"] < s2["accuracy"]: return "S2_BETTER"
    return "NO_CLEAR_DIFFERENCE"


def evidence_gate(*, task_classes: dict, s1_changes: int, s2_changes: int, total_records: int) -> tuple[str, str]:
    """Apply the preregistered evidence rule without subjective scoring."""
    joint_better = any(value == "JOINT_BETTER" for value in task_classes.values())
    s2_better = sum(value == "S2_BETTER" for value in task_classes.values())
    both_modalities_affect = s1_changes > 0 and s2_changes > 0
    if joint_better and both_modalities_affect and s2_better == 0:
        return "OPTICAL_SAR_SCIENTIFIC_GAIN_SUPPORTED", "CONTROLLED_OPTICAL_SAR_ROUTE_CANDIDATE"
    if both_modalities_affect:
        return "OPTICAL_SAR_MULTIMODAL_USE_SUPPORTED_NO_CLEAR_GAIN", "MANUAL_EXPERIMENTAL_OPTICAL_SAR_MODE"
    if s2_better or s1_changes == 0:
        return "OPTICAL_SAR_TECHNICALLY_VALID_S2_DOMINANT", "KEEP_S2_DEFAULT_AND_RETAIN_FUSION_FOR_RESEARCH_ONLY"
    return "OPTICAL_SAR_NOT_SCIENTIFICALLY_JUSTIFIED", "DISABLE_OPTICAL_SAR_USER_PATH"
