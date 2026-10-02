"""Split-safe benchmark record normalization without model execution.

This small contract is deliberately independent of application inference and
never opens benchmark files on its own. It rejects test records by policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class BenchmarkAdmissionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark: str
    record_id: str
    image_path: str
    question: str | None
    reference_answer: str | None
    task: str
    split: str
    source: str
    provenance: Mapping[str, Any]

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


def normalize_development_record(raw: Mapping[str, Any]) -> BenchmarkRecord:
    """Normalize an already-admitted train/validation record; test is forbidden."""
    split = str(raw.get("split") or "").lower()
    if split == "test":
        raise BenchmarkAdmissionError("TEST_SPLIT_REJECTED", "Benchmark test records cannot enter development evaluation.")
    if split not in {"train", "validation", "val", "development"}:
        raise BenchmarkAdmissionError("BENCHMARK_SPLIT_REQUIRED", "A benchmark record requires an explicit development-safe split.")
    benchmark, record_id, image_path, task = (str(raw.get(key) or "").strip() for key in ("benchmark", "record_id", "image_path", "task"))
    if not all((benchmark, record_id, image_path, task)):
        raise BenchmarkAdmissionError("BENCHMARK_RECORD_INCOMPLETE", "benchmark, record_id, image_path, and task are required.")
    return BenchmarkRecord(benchmark=benchmark, record_id=record_id, image_path=image_path,
        question=str(raw["question"]) if raw.get("question") is not None else None,
        reference_answer=str(raw["reference_answer"]) if raw.get("reference_answer") is not None else None,
        task=task, split="validation" if split == "val" else split,
        source=str(raw.get("source") or "").strip() or "UNSPECIFIED",
        provenance=dict(raw.get("provenance") or {}))
