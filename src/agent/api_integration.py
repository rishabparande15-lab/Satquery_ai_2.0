"""HTTP-safe controller facade: no model paths or exceptions leave this boundary."""
from __future__ import annotations
from typing import Any, Mapping
from .controller import AgentController
from .query_types import AnalysisRequest, ExecutionStatus
from .workflow import summarize, deterministic_report
from .scene_resolver import resolve


def _safe(value: Any) -> Any:
    if hasattr(value, "value"): return value.value
    if hasattr(value, "__dict__"): return {key: _safe(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items() if "path" not in str(key).lower() and "token" not in str(key).lower()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    return value


def analyze_controller_request(payload: Mapping[str, Any], controller: AgentController | None = None) -> dict[str, Any]:
    query = payload.get("query")
    inputs = payload.get("inputs", ())
    if payload.get('scene') is not None: inputs=[resolve(payload['scene'])]
    if not isinstance(inputs, list): raise ValueError("inputs must be a list")
    normalized = tuple({key: value for key, value in item.items() if key in {"id", "modality", "source_reference"}} for item in inputs if isinstance(item, Mapping))
    if len(normalized) != len(inputs): raise ValueError("each input must be an object")
    result = (controller or AgentController()).analyze(AnalysisRequest(str(query or ""), normalized, dict(payload.get("temporal") or {})))
    blocked_reason = result.final_answer.text if result.capability_status == ExecutionStatus.BLOCKED else None
    return {"execution_id": result.execution_id, "status": result.capability_status.value,
            "task_type": result.plan.understanding.task_type.value, "final_answer": _safe(result.final_answer),
            "capability_status": result.capability_status.value, "blocked_reason": blocked_reason,
            "evidence": _safe(result.evidence), "provenance": _safe(result.provenance),
            "execution_trace": _safe(result.trace), "audit_summary": _safe(summarize(result)["audit"]),
            "execution_summary": _safe(summarize(result)["execution"]), "evidence_summary": _safe(summarize(result)["evidence_summary"]),
            "report": _safe(deterministic_report(result))}
