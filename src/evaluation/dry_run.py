"""Synthetic-only evaluation-system validation. Never use this module for a benchmark run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.eo_vlm.grounding_contracts import ImageDimensions, MaskReference, Polygon, validate_geometry
from src.spatial_contract import BBox, CoordinateSpace, Point, GeometryStatus

from .contracts import (BenchmarkPrediction, BenchmarkSpec, CompatibilityStatus, MetricResult,
                        ModelCompatibility, VerificationStatus)
from .provenance import canonical_result_bytes

DRY_RUN_MARKERS = {"run_kind": "DRY_RUN", "fixture_kind": "SYNTHETIC_FIXTURE", "result_status": "NOT_BENCHMARK_RESULT"}


@dataclass(frozen=True)
class DryRunFixture:
    fixture_id: str
    benchmark_id: str
    split: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.benchmark_id or not self.split:
            raise ValueError("dry-run fixture requires fixture_id, benchmark_id, and split")
        if not isinstance(self.payload, Mapping):
            raise ValueError("dry-run fixture payload must be a mapping")


def synthetic_fixtures() -> dict[str, DryRunFixture]:
    return {
        "RSVQA_BINARY": DryRunFixture("fixture-rsvqa-binary", "RSVQA_BINARY", "test", {"image": "synthetic-rgb", "question": "synthetic?", "target": "yes"}),
        "RSVQA_MULTIPLE_CHOICE": DryRunFixture("fixture-rsvqa-mcq", "RSVQA_MULTIPLE_CHOICE", "test", {"image": "synthetic-rgb", "question": "synthetic?", "choices": ("a", "b"), "target": "a"}),
        "RSVQA_OPEN_ENDED": DryRunFixture("fixture-rsvqa-open", "RSVQA_OPEN_ENDED", "test", {"image": "synthetic-rgb", "question": "synthetic?", "target": "synthetic answer"}),
        "VRSBENCH_GROUNDING": DryRunFixture("fixture-vrsbench", "VRSBENCH_GROUNDING", "test", {"image": "synthetic-rgb", "query": "synthetic object", "dimensions": (10, 10), "target": BBox(1, 1, 4, 4, CoordinateSpace.PIXEL)}),
        "CDVQA_CHANGE_VQA": DryRunFixture("fixture-cdvqa", "CDVQA_CHANGE_VQA", "test", {"t1": "synthetic-t1", "t2": "synthetic-t2", "temporal_order": "T1_BEFORE_T2", "question": "synthetic change?", "target": "yes"}),
    }


def validate_fixture(fixture: DryRunFixture, spec: BenchmarkSpec) -> None:
    if fixture.benchmark_id != spec.benchmark_id:
        raise ValueError("BENCHMARK_MISMATCH")
    if fixture.split not in spec.splits:
        raise ValueError("SPLIT_MISMATCH")
    payload = fixture.payload
    if fixture.benchmark_id.startswith("RSVQA"):
        if not payload.get("image") or not payload.get("question") or not payload.get("target"):
            raise ValueError("RSVQA_FIXTURE_INVALID")
        if fixture.benchmark_id == "RSVQA_MULTIPLE_CHOICE" and payload.get("target") not in payload.get("choices", ()):
            raise ValueError("RSVQA_MULTIPLE_CHOICE_INVALID")
    elif fixture.benchmark_id == "VRSBENCH_GROUNDING":
        dims = payload.get("dimensions")
        geometry = payload.get("target")
        if not payload.get("image") or not payload.get("query") or not isinstance(dims, tuple) or len(dims) != 2:
            raise ValueError("GROUNDING_FIXTURE_INVALID")
        if validate_geometry(geometry, ImageDimensions(*dims)) is not GeometryStatus.VALID:
            raise ValueError("GROUNDING_GEOMETRY_INVALID")
    elif fixture.benchmark_id == "CDVQA_CHANGE_VQA":
        if not payload.get("t1"):
            raise ValueError("TEMPORAL_T1_MISSING")
        if not payload.get("t2"):
            raise ValueError("TEMPORAL_T2_MISSING")
        if payload.get("temporal_order") in (None, "UNKNOWN"):
            raise ValueError("TEMPORAL_ORDER_UNKNOWN")
        if not payload.get("question") or not payload.get("target"):
            raise ValueError("TEMPORAL_FIXTURE_INVALID")
    else:
        raise ValueError("UNSUPPORTED_DRY_RUN_BENCHMARK")


def validate_prediction(prediction: BenchmarkPrediction, spec: BenchmarkSpec, fixture: DryRunFixture) -> None:
    if prediction.benchmark_id != spec.benchmark_id or prediction.benchmark_id != fixture.benchmark_id:
        raise ValueError("BENCHMARK_MISMATCH")
    if prediction.sample_id != fixture.fixture_id:
        raise ValueError("SAMPLE_ID_MISMATCH")
    if prediction.split != fixture.split:
        raise ValueError("SPLIT_MISMATCH")
    if prediction.validation_status is not VerificationStatus.VERIFIED:
        raise ValueError("PREDICTION_NOT_VERIFIED")
    if not prediction.provenance or prediction.provenance.get("run_kind") != "DRY_RUN":
        raise ValueError("PROVENANCE_INVALID")
    if prediction.prediction is None or (isinstance(prediction.prediction, str) and not prediction.prediction.strip()):
        raise ValueError("PREDICTION_MALFORMED")
    target = fixture.payload.get("target")
    if prediction.target != target or prediction.target_reference != f"fixture:{fixture.fixture_id}:target":
        raise ValueError("TARGET_LINKAGE_INVALID")
    if spec.benchmark_id == "RSVQA_BINARY" and prediction.prediction not in {"yes", "no"}:
        raise ValueError("BINARY_PREDICTION_INVALID")
    if spec.benchmark_id == "RSVQA_MULTIPLE_CHOICE" and prediction.prediction not in fixture.payload["choices"]:
        raise ValueError("MULTIPLE_CHOICE_PREDICTION_INVALID")


def dry_run_model_compatibility(capability_id: str, benchmark_id: str) -> ModelCompatibility:
    compatible = capability_id == "QWEN_RGB_DRY_RUN" and benchmark_id.startswith("RSVQA")
    status = CompatibilityStatus.COMPATIBLE if compatible else CompatibilityStatus.BLOCKED
    return ModelCompatibility(capability_id, benchmark_id, status, "synthetic structural compatibility only" if compatible else "capability is not admitted for this benchmark", "synthetic-model" if compatible else None, "dry-run" if compatible else None, "synthetic-adapter" if compatible else None, "dry-run" if compatible else None)


class DryRunEvaluator:
    """Computes only an explicitly synthetic RSVQA accuracy check."""
    evaluator_id = "satquery_dry_run_structural_evaluator"
    evaluator_version = "phase3n1"

    def evaluate(self, prediction: BenchmarkPrediction, spec: BenchmarkSpec, fixture: DryRunFixture) -> tuple[MetricResult, ...]:
        validate_fixture(fixture, spec)
        validate_prediction(prediction, spec, fixture)
        if "accuracy" not in spec.metrics:
            return ()
        value = float(prediction.prediction == fixture.payload["target"])
        return (MetricResult("accuracy", value, prediction.split, 1, self.evaluator_version,
                             {**DRY_RUN_MARKERS, "fixture_id": fixture.fixture_id, "benchmark_id": spec.benchmark_id}),)


def dry_run_artifacts(fixture: DryRunFixture, prediction: BenchmarkPrediction, metrics: tuple[MetricResult, ...]) -> dict[str, bytes]:
    """Return deterministic in-memory artifacts. Callers decide whether a test temp path writes them."""
    common = {**DRY_RUN_MARKERS, "benchmark_id": fixture.benchmark_id, "fixture_id": fixture.fixture_id, "split": fixture.split, "execution_id": prediction.execution_id, "model_id": prediction.model_id, "model_revision": prediction.model_revision, "evaluator_version": DryRunEvaluator.evaluator_version, "provenance_status": prediction.validation_status.value}
    payloads: dict[str, Any] = {
        "evaluation_config.json": common,
        "dry_run_predictions.jsonl": {**common, "prediction": prediction.to_dict()},
        "dry_run_metrics.json": {**common, "metrics": [item.to_dict() for item in metrics]},
        "dry_run_provenance.json": {**common, "provenance": dict(prediction.provenance)},
        "dry_run_report.md": "# DRY_RUN\n\nSYNTHETIC_FIXTURE\n\nNOT_BENCHMARK_RESULT\n",
    }
    return {name: value.encode("utf-8") if isinstance(value, str) else canonical_result_bytes(value) + b"\n" for name, value in payloads.items()}


def synthetic_grounding_geometries() -> dict[str, Any]:
    dimensions = ImageDimensions(10, 10)
    return {
        "point": Point(1, 1, CoordinateSpace.PIXEL),
        "bbox": BBox(1, 1, 4, 4, CoordinateSpace.PIXEL),
        "mask": MaskReference("synthetic-mask", dimensions, CoordinateSpace.PIXEL, {"run_kind": "DRY_RUN"}),
        "invalid": BBox(4, 4, 1, 1, CoordinateSpace.PIXEL),
        "coordinate_mismatch": BBox(1, 1, 12, 12, CoordinateSpace.PIXEL),
    }
