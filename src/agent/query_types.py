"""Typed, deterministic request interpretation for the agent controller."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TaskType(str, Enum):
    SCIENTIFIC_ANALYSIS = "SCIENTIFIC_ANALYSIS"
    RGB_VISUAL_REASONING = "RGB_VISUAL_REASONING"
    S2_VQA = "S2_VQA"
    S2_CAPTIONING = "S2_CAPTIONING"
    SAR_VQA = "SAR_VQA"
    OPTICAL_SAR_REASONING = "OPTICAL_SAR_REASONING"
    TEMPORAL_REASONING = "TEMPORAL_REASONING"
    CHANGE_VQA = "CHANGE_VQA"
    TEMPORAL_RGB_EXECUTION = "TEMPORAL_RGB_EXECUTION"
    GROUNDING = "GROUNDING"
    REPORT_GENERATION = "REPORT_GENERATION"


class ExecutionStatus(str, Enum):
    READY = "READY"
    EXECUTED = "EXECUTED"
    BLOCKED = "BLOCKED"
    DEFERRED = "DEFERRED"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"
    CONFLICT = "CONFLICT"
    NOT_VERIFIED = "NOT_VERIFIED"


@dataclass(frozen=True)
class AnalysisRequest:
    query: str
    inputs: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(self.inputs, tuple) or any(not isinstance(item, Mapping) for item in self.inputs):
            raise ValueError("inputs must be a tuple of mappings")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")


@dataclass(frozen=True)
class QueryUnderstanding:
    raw_query: str
    task_type: TaskType
    modality: str | None
    image_count: int
    temporal_required: bool
    grounding_required: bool
    scientific_required: bool
    requested_output_type: str
    required_capabilities: tuple[TaskType, ...]
    validation_status: ExecutionStatus = ExecutionStatus.READY
    rationale: tuple[str, ...] = ()
