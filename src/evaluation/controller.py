"""Controller integration boundary: benchmark requests never fall back to VLM generation."""
from dataclasses import dataclass

from .readiness import EvaluationReadiness


@dataclass(frozen=True)
class BenchmarkEvaluationDecision:
    status: str
    fallback: str
    reasons: tuple[str, ...]


def decide_benchmark_execution(readiness: EvaluationReadiness) -> BenchmarkEvaluationDecision:
    if not readiness.ready:
        return BenchmarkEvaluationDecision("EVALUATION_BLOCKED", "NONE", readiness.reasons)
    return BenchmarkEvaluationDecision("EVALUATION_READY", "NONE", ())
