"""Audit trace for every controller decision, including non-execution."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from .query_types import ExecutionStatus


@dataclass(frozen=True)
class TraceStep:
    name: str
    status: ExecutionStatus
    input_references: tuple[str, ...] = ()
    output_references: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


@dataclass(frozen=True)
class ExecutionTrace:
    execution_id: str
    timestamp: str
    steps: tuple[TraceStep, ...]

    @classmethod
    def start(cls, execution_id: str) -> "ExecutionTrace":
        return cls(execution_id, datetime.now(timezone.utc).isoformat(), ())

    def append(self, step: TraceStep) -> "ExecutionTrace":
        return ExecutionTrace(self.execution_id, self.timestamp, self.steps + (step,))
