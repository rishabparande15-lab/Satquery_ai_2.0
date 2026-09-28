"""Typed, model-neutral contracts for future benchmark evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

UNKNOWN = "UNKNOWN"


class BenchmarkStatus(str, Enum):
    DATASET_UNAVAILABLE = "DATASET_UNAVAILABLE"
    DATASET_UNVERIFIED = "DATASET_UNVERIFIED"
    DATASET_READY = "DATASET_READY"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_READY = "MODEL_READY"
    EVALUATOR_AVAILABLE = "EVALUATOR_AVAILABLE"
    EVALUATOR_UNAVAILABLE = "EVALUATOR_UNAVAILABLE"
    EVALUATION_READY = "EVALUATION_READY"
    RESULT_AVAILABLE = "RESULT_AVAILABLE"
    EVALUATION_BLOCKED = "EVALUATION_BLOCKED"


class RealEvaluationStatus(str, Enum):
    READY_FOR_REAL_EVALUATION = "READY_FOR_REAL_EVALUATION"
    PARTIALLY_READY = "PARTIALLY_READY"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    BLOCKED = "BLOCKED"


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNVERIFIED = "UNVERIFIED"
    BLOCKED = "BLOCKED"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    task_type: str
    modality: str
    dataset_revision: str = UNKNOWN
    benchmark_revision: str = UNKNOWN
    splits: tuple[str, ...] = ("test",)
    expected_input: Mapping[str, Any] = field(default_factory=dict)
    expected_target: Mapping[str, Any] = field(default_factory=dict)
    evaluator_id: str = UNKNOWN
    metrics: tuple[str, ...] = ()
    required_provenance: tuple[str, ...] = ()
    license_requirements: tuple[str, ...] = ()
    leakage_requirements: tuple[str, ...] = ()
    expected_output_format: Mapping[str, Any] = field(default_factory=dict)
    dataset_status: BenchmarkStatus = BenchmarkStatus.DATASET_UNVERIFIED

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "task_type", "modality", "dataset_revision", "benchmark_revision", "evaluator_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not self.splits or any(not isinstance(item, str) or not item for item in self.splits):
            raise ValueError("splits must contain named splits")
        object.__setattr__(self, "expected_input", _mapping(self.expected_input, "expected_input"))
        object.__setattr__(self, "expected_target", _mapping(self.expected_target, "expected_target"))
        object.__setattr__(self, "expected_output_format", _mapping(self.expected_output_format, "expected_output_format"))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset_status"] = self.dataset_status.value
        return value


@dataclass(frozen=True)
class DatasetReadiness:
    source: str = UNKNOWN
    dataset_revision: str = UNKNOWN
    benchmark_revision: str = UNKNOWN
    image_availability: VerificationStatus = VerificationStatus.UNVERIFIED
    annotation_availability: VerificationStatus = VerificationStatus.UNVERIFIED
    sample_linkage: VerificationStatus = VerificationStatus.UNVERIFIED
    license_status: VerificationStatus = VerificationStatus.UNVERIFIED
    provenance_status: VerificationStatus = VerificationStatus.UNVERIFIED
    leakage_status: VerificationStatus = VerificationStatus.UNVERIFIED
    modality_compatibility: VerificationStatus = VerificationStatus.UNVERIFIED
    split_manifest: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source", "dataset_revision", "benchmark_revision"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "split_manifest", {str(key): tuple(value) for key, value in _mapping(self.split_manifest, "split_manifest").items()})


@dataclass(frozen=True)
class ModelCompatibility:
    capability_id: str
    benchmark_id: str
    status: CompatibilityStatus
    reason: str
    model_id: str | None = None
    model_revision: str | None = None
    adapter_id: str | None = None
    adapter_revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("capability_id", "benchmark_id", "reason"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))


@dataclass(frozen=True)
class BenchmarkPrediction:
    benchmark_id: str
    sample_id: str
    prediction: Any
    split: str
    execution_id: str
    model_id: str
    model_revision: str
    adapter_id: str
    adapter_revision: str
    provenance: Mapping[str, Any]
    validation_status: VerificationStatus = VerificationStatus.VERIFIED
    target: Any | None = None
    target_reference: str | None = None

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "sample_id", "split", "execution_id", "model_id", "model_revision", "adapter_id", "adapter_revision"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))
        if self.target is not None and not self.target_reference:
            raise ValueError("target requires an authoritative target_reference")
        if self.target_reference is not None:
            object.__setattr__(self, "target_reference", _nonempty(self.target_reference, "target_reference"))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["validation_status"] = self.validation_status.value
        return value


@dataclass(frozen=True)
class MetricResult:
    metric_name: str
    value: float
    split: str
    sample_count: int
    evaluator_version: str
    provenance: Mapping[str, Any]
    unit: str | None = None
    confidence_interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("metric_name", "split", "evaluator_version"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("metric value must be numeric")
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))
        if self.confidence_interval is not None and (len(self.confidence_interval) != 2 or self.confidence_interval[0] > self.confidence_interval[1]):
            raise ValueError("confidence_interval must be ordered lower/upper bounds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
