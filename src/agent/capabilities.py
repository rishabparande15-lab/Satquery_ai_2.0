"""An explicit registry; registration is not a claim that a model was run."""
from dataclasses import dataclass
from enum import Enum
from .query_types import TaskType


class CapabilityState(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    BLOCKED = "BLOCKED"
    DEFERRED = "DEFERRED"
    UNSUPPORTED = "UNSUPPORTED"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class Capability:
    task_type: TaskType
    state: CapabilityState
    tool_name: str | None
    reason: str
    representations: tuple[str, ...] = ()


class CapabilityRegistry:
    def __init__(self, capabilities: tuple[Capability, ...]):
        self._items = {item.task_type: item for item in capabilities}
        if len(self._items) != len(capabilities):
            raise ValueError("only one capability may be registered per task type")

    def get(self, task_type: TaskType) -> Capability:
        return self._items[task_type]

    def all(self) -> tuple[Capability, ...]:
        return tuple(self._items[item] for item in sorted(self._items, key=lambda value: value.value))


def default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry((
        Capability(TaskType.SCIENTIFIC_ANALYSIS, CapabilityState.IMPLEMENTED, "scientific_predictor", "Frozen validated Pipeline 3 scientific predictor.", ("physical_62d", "joint_croma_gap_768d", "hybrid_830d")),
        Capability(TaskType.RGB_VISUAL_REASONING, CapabilityState.IMPLEMENTED, "qwen_rgb", "Qwen RGB processing and generation validated.", ("RGB",)),
        Capability(TaskType.S2_VQA, CapabilityState.EXPERIMENTAL, "qwen_s2", "Learned S2 language path is a validated PoC, not a production benchmark claim.", ("learned_s2_representation",)),
        Capability(TaskType.S2_CAPTIONING, CapabilityState.EXPERIMENTAL, "qwen_s2", "Learned S2 language path is a validated PoC, not a production benchmark claim.", ("learned_s2_representation",)),
        Capability(TaskType.SAR_VQA, CapabilityState.BLOCKED, None, "SAR language supervision/training is blocked."),
        Capability(TaskType.OPTICAL_SAR_REASONING, CapabilityState.BLOCKED, None, "Joint optical-SAR supervision and fusion are blocked."),
        Capability(TaskType.TEMPORAL_REASONING, CapabilityState.BLOCKED, None, "Temporal learned reasoning/training is blocked; two images are not change evidence."),
        Capability(TaskType.CHANGE_VQA, CapabilityState.BLOCKED, None, "Change-VQA training is blocked."),
        Capability(TaskType.TEMPORAL_RGB_EXECUTION, CapabilityState.IMPLEMENTED, "temporal_rgb", "Validated RGB-pair execution only; not learned temporal/change reasoning.", ("RGB_T1", "RGB_T2")),
        Capability(TaskType.GROUNDING, CapabilityState.BLOCKED, None, "Grounding has a contract only; no learned grounding model is available."),
        Capability(TaskType.REPORT_GENERATION, CapabilityState.IMPLEMENTED, "report_assembler", "Deterministic report assembly over collected evidence."),
    ))
