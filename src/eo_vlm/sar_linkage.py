"""Fail-closed S1/BigEarthNet.txt linkage and label-applicability helpers."""
from __future__ import annotations

from typing import Mapping


SAR_UNKNOWN = "SAR_UNKNOWN"


def require_exact_s1_link(annotation: Mapping[str, object], source: Mapping[str, object]) -> None:
    """Require both independently-authoritative BigEarthNet identities."""
    if annotation.get("image_id") != source.get("patch_id"):
        raise ValueError("IMAGE_LINKAGE_BLOCKED: patch/image identity mismatch")
    if annotation.get("sar_identity") != source.get("s1_name"):
        raise ValueError("IMAGE_LINKAGE_BLOCKED: S1 identity mismatch")


def sar_annotation_applicability(annotation: Mapping[str, object]) -> tuple[str, str]:
    """Conservative decision until publisher supplies SAR-only supervision evidence.

    BigEarthNet.txt proves paired/co-registered S1/S2 source identity but its
    records do not declare that a given question/answer is valid for SAR-only
    observation.  Geographic linkage is deliberately insufficient.
    """
    if annotation.get("task_type") not in {"binary_qa", "multiple_choice_qa"}:
        raise ValueError("annotation is not a visual-VQA candidate")
    return SAR_UNKNOWN, "publisher record has no SAR-only label applicability declaration"
