"""Deterministic schema declarations for artifacts produced only by real runs."""
from hashlib import sha256
from typing import Any, Mapping

from src.annotation_foundation import canonical_bytes

RESULT_ARTIFACTS = ("benchmark_manifest.json", "evaluation_config.json", "predictions.jsonl", "metrics.json", "provenance.json", "evaluation_report.md")


def canonical_result_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_bytes(dict(value))


def result_artifact_hash(value: Mapping[str, Any]) -> str:
    return sha256(canonical_result_bytes(value)).hexdigest()


def result_artifact_schema() -> dict[str, tuple[str, ...]]:
    return {
        "benchmark_manifest.json": ("benchmark_id", "benchmark_revision", "dataset_revision", "split", "sample_manifest"),
        "evaluation_config.json": ("model_revision", "adapter_revision", "preprocessing_version", "random_seed", "configuration"),
        "predictions.jsonl": ("benchmark_id", "sample_id", "prediction", "split", "execution_id", "provenance", "validation_status"),
        "metrics.json": ("metric_name", "value", "split", "sample_count", "evaluator_version", "provenance"),
        "provenance.json": ("prediction_artifact_hash", "result_artifact_hash"),
        "evaluation_report.md": ("status", "blocked_reasons"),
    }
