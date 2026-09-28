"""Fail-closed dataset and run readiness checks; they never discard bad samples."""
from dataclasses import dataclass

from .contracts import (BenchmarkSpec, BenchmarkStatus, CompatibilityStatus, DatasetReadiness,
                        ModelCompatibility, RealEvaluationStatus, VerificationStatus, UNKNOWN)


@dataclass(frozen=True)
class EvaluationReadiness:
    status: BenchmarkStatus
    reasons: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status is BenchmarkStatus.EVALUATION_READY


@dataclass(frozen=True)
class RealEvaluationDecision:
    status: RealEvaluationStatus
    reasons: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.status is RealEvaluationStatus.READY_FOR_REAL_EVALUATION


def check_dataset_readiness(spec: BenchmarkSpec, dataset: DatasetReadiness, *, split: str, sample_ids: tuple[str, ...], train_sample_ids: tuple[str, ...] = (), model: ModelCompatibility | None = None, evaluator_available: bool = False) -> EvaluationReadiness:
    reasons: list[str] = []
    if spec.dataset_status is BenchmarkStatus.DATASET_UNAVAILABLE:
        reasons.append("DATASET_UNAVAILABLE")
    for name in ("source", "dataset_revision", "benchmark_revision"):
        if getattr(dataset, name) == UNKNOWN:
            reasons.append(f"{name.upper()}_UNKNOWN")
    if dataset.benchmark_revision != spec.benchmark_revision:
        reasons.append("BENCHMARK_REVISION_MISMATCH")
    for name in ("image_availability", "annotation_availability", "sample_linkage", "license_status", "provenance_status", "leakage_status", "modality_compatibility"):
        if getattr(dataset, name) is not VerificationStatus.VERIFIED:
            reasons.append(f"{name.upper()}_{getattr(dataset, name).value}")
    if not split or split == UNKNOWN or split not in spec.splits or split not in dataset.split_manifest:
        reasons.append("SPLIT_UNKNOWN_OR_MISSING")
    if not sample_ids:
        reasons.append("SAMPLE_SELECTION_EMPTY")
    if len(set(sample_ids)) != len(sample_ids):
        reasons.append("DUPLICATE_SAMPLE_ID")
    if set(sample_ids).intersection(train_sample_ids):
        reasons.append("TRAIN_TEST_OVERLAP")
    if model is None or model.status is not CompatibilityStatus.COMPATIBLE:
        reasons.append("MODEL_UNAVAILABLE_OR_INCOMPATIBLE")
    if not evaluator_available:
        reasons.append("EVALUATOR_UNAVAILABLE")
    return EvaluationReadiness(BenchmarkStatus.EVALUATION_READY if not reasons else BenchmarkStatus.EVALUATION_BLOCKED, tuple(reasons))


def real_evaluation_gate(spec: BenchmarkSpec, dataset: DatasetReadiness, *, split: str,
                         sample_ids: tuple[str, ...], train_sample_ids: tuple[str, ...] = (),
                         model: ModelCompatibility | None = None, evaluator_available: bool = False,
                         prediction_compatible: bool = False) -> RealEvaluationDecision:
    """Answer whether a real benchmark may execute; every unknown is fail-closed."""
    if spec.benchmark_id == "ISRO_SAC_HIDDEN_FINAL" and (
        spec.dataset_status is BenchmarkStatus.DATASET_UNVERIFIED or dataset.source == UNKNOWN
    ):
        return RealEvaluationDecision(RealEvaluationStatus.UNVERIFIED, ("HIDDEN_DATASET_UNVERIFIED",))
    data_probe = ModelCompatibility("readiness_probe", spec.benchmark_id, CompatibilityStatus.COMPATIBLE, "data-only readiness probe")
    data = check_dataset_readiness(spec, dataset, split=split, sample_ids=sample_ids,
                                   train_sample_ids=train_sample_ids, model=data_probe,
                                   evaluator_available=True)
    data_reasons = tuple(reason for reason in data.reasons if reason not in {"MODEL_UNAVAILABLE_OR_INCOMPATIBLE", "EVALUATOR_UNAVAILABLE"})
    if data_reasons:
        return RealEvaluationDecision(RealEvaluationStatus.BLOCKED, data_reasons)
    reasons: list[str] = []
    if model is None or model.status is not CompatibilityStatus.COMPATIBLE:
        reasons.append("MODEL_UNAVAILABLE_OR_INCOMPATIBLE")
    if not evaluator_available:
        reasons.append("EVALUATOR_UNAVAILABLE")
    if not prediction_compatible:
        reasons.append("PREDICTION_FORMAT_INCOMPATIBLE")
    return RealEvaluationDecision(RealEvaluationStatus.READY_FOR_REAL_EVALUATION if not reasons else RealEvaluationStatus.PARTIALLY_READY, tuple(reasons))
