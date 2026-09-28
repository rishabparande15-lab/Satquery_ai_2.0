"""Fail-closed benchmark evaluation infrastructure; no benchmark execution."""

from .benchmark_specs import BENCHMARK_REGISTRY, get_benchmark_spec
from .contracts import BenchmarkPrediction, MetricResult
from .readiness import check_dataset_readiness
from .rsvqa import RSVQADatasetAcceptance

__all__ = ("BENCHMARK_REGISTRY", "BenchmarkPrediction", "MetricResult", "RSVQADatasetAcceptance", "check_dataset_readiness", "get_benchmark_spec")
