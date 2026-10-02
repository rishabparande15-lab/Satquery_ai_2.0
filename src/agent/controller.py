"""Dependency-injected, fail-closed controller; no model implementation lives here."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4
from .answer_assembler import AnswerAssembler, AnswerAssembly
from .capabilities import CapabilityRegistry, CapabilityState, default_capability_registry
from .evidence import Evidence
from .execution_trace import ExecutionTrace, TraceStep
from .planner import DeterministicPlanner, ExecutionPlan
from .query_types import AnalysisRequest, ExecutionStatus, TaskType
from .reconciler import EvidenceReconciler, Reconciliation


@dataclass(frozen=True)
class ToolResult:
    output: Mapping[str, Any]
    evidence: tuple[Evidence, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


class Tool(ABC):
    name: str

    @abstractmethod
    def execute(self, request: AnalysisRequest, *, representations: tuple[str, ...]) -> ToolResult:
        """Execute a concrete validated tool; implementations own model details."""


@dataclass(frozen=True)
class AnalysisResult:
    execution_id: str
    plan: ExecutionPlan
    tool_results: tuple[ToolResult, ...]
    evidence: tuple[Evidence, ...]
    final_answer: AnswerAssembly
    capability_status: ExecutionStatus
    provenance: Mapping[str, Any]
    audit_summary: Mapping[str, Any]
    trace: ExecutionTrace


class AgentController:
    def __init__(self, tools: Mapping[str, Tool] | None = None, *, registry: CapabilityRegistry | None = None) -> None:
        self._tools = dict(tools or {})
        self._registry = registry or default_capability_registry()
        self._planner = DeterministicPlanner()
        self._reconciler = EvidenceReconciler()
        self._assembler = AnswerAssembler()

    @staticmethod
    def _representations(task: TaskType) -> tuple[str, ...]:
        if task == TaskType.SCIENTIFIC_ANALYSIS:
            return ("physical_62d", "joint_croma_gap_768d", "hybrid_830d")
        if task in {TaskType.SINGLE_IMAGE_VQA, TaskType.S2_VQA, TaskType.S2_CAPTIONING}:
            return ("learned_s2_representation",)
        if task == TaskType.OPTICAL_SAR_ANALYSIS:
            return ("S1", "S2", "CROMA_JOINT")
        if task == TaskType.RGB_VISUAL_REASONING:
            return ("RGB",)
        if task == TaskType.TEMPORAL_RGB_EXECUTION:
            return ("RGB_T1", "RGB_T2")
        if task == TaskType.TEMPORAL_CHANGE_DESCRIPTION:
            return ("RGB_T1", "RGB_T2")
        return ()

    @staticmethod
    def _validate(request: AnalysisRequest, task: TaskType) -> None:
        if task != TaskType.REPORT_GENERATION and not request.inputs:
            raise ValueError("at least one declared input is required")
        for item in request.inputs:
            if not item.get("id"):
                raise ValueError("each input requires a stable id")
            if task != TaskType.REPORT_GENERATION and not item.get("modality"):
                raise ValueError("each input requires an explicit modality")
        if task == TaskType.TEMPORAL_CHANGE_DESCRIPTION:
            if len(request.inputs) != 2:
                raise ValueError("TEMPORAL_CHANGE_DESCRIPTION requires exactly T1 and T2 inputs")
            if any(str(item.get("modality", "")).lower() in {"sar", "s1", "optical_sar"} for item in request.inputs):
                raise ValueError("TEMPORAL_CHANGE_DESCRIPTION accepts an optical PRE/POST pair, not an optical-SAR pair")
        # Existing adapters/contracts remain authoritative for format, shape, dtype,
        # finite values, CRS, temporal-pair, and grounding geometry validation.

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        execution_id = str(uuid4())
        trace = ExecutionTrace.start(execution_id)
        understanding = self._planner.understand(request)
        trace = trace.append(TraceStep("query_interpretation", ExecutionStatus.EXECUTED, provenance={"rationale": understanding.rationale, "task_type": understanding.task_type.value}))
        capability = self._registry.get(understanding.task_type)
        representations = self._representations(understanding.task_type)
        plan = self._planner.plan(request, capability.tool_name, representations)
        try:
            self._validate(request, understanding.task_type)
            trace = trace.append(TraceStep("validation", ExecutionStatus.EXECUTED, input_references=tuple(str(item["id"]) for item in request.inputs)))
        except ValueError as exc:
            trace = trace.append(TraceStep("validation", ExecutionStatus.FAILED, provenance={"reason": str(exc)}))
            raise
        trace = trace.append(TraceStep("representation_selection", ExecutionStatus.EXECUTED, output_references=representations, provenance={"scientific_predictor_only": "hybrid_830d" in representations}))
        if capability.state == CapabilityState.BLOCKED:
            reason = f"{understanding.task_type.value}_BLOCKED: {capability.reason} Fallback: NONE."
            trace = trace.append(TraceStep("tool_selection", ExecutionStatus.BLOCKED, provenance={"reason": reason, "fallback": "NONE"}))
            trace = trace.append(TraceStep("tool_execution", ExecutionStatus.BLOCKED, provenance={"reason": reason}))
            trace = trace.append(TraceStep("evidence_collection", ExecutionStatus.BLOCKED, provenance={"reason": reason}))
            reconciliation = self._reconciler.reconcile(())
            trace = trace.append(TraceStep("reconciliation", ExecutionStatus.BLOCKED, provenance={"reason": reason}))
            answer = self._assembler.assemble(reconciliation, reason)
            trace = trace.append(TraceStep("answer_generation", ExecutionStatus.BLOCKED, provenance={"reason": reason}))
            return self._result(execution_id, plan, (), reconciliation, answer, ExecutionStatus.BLOCKED, trace, request)
        if capability.tool_name is None or capability.tool_name not in self._tools:
            reason = f"{understanding.task_type.value}_NOT_VERIFIED: no configured validated tool instance. Fallback: NONE."
            trace = trace.append(TraceStep("tool_selection", ExecutionStatus.NOT_VERIFIED, provenance={"reason": reason, "fallback": "NONE"}))
            trace = trace.append(TraceStep("tool_execution", ExecutionStatus.NOT_VERIFIED, provenance={"reason": reason}))
            trace = trace.append(TraceStep("evidence_collection", ExecutionStatus.NOT_VERIFIED, provenance={"reason": reason}))
            reconciliation = self._reconciler.reconcile(())
            trace = trace.append(TraceStep("reconciliation", ExecutionStatus.NOT_VERIFIED, provenance={"reason": reason}))
            answer = self._assembler.assemble(reconciliation, reason)
            trace = trace.append(TraceStep("answer_generation", ExecutionStatus.NOT_VERIFIED, provenance={"reason": reason}))
            return self._result(execution_id, plan, (), reconciliation, answer, ExecutionStatus.NOT_VERIFIED, trace, request)
        tool = self._tools[capability.tool_name]
        trace = trace.append(TraceStep("tool_selection", ExecutionStatus.EXECUTED, output_references=(tool.name,)))
        started = perf_counter()
        try:
            tool_result = tool.execute(request, representations=representations)
        except Exception as exc:
            trace = trace.append(TraceStep("tool_execution", ExecutionStatus.FAILED, provenance={"tool": tool.name, "reason": str(exc)}, duration_ms=(perf_counter() - started) * 1000))
            reconciliation = self._reconciler.reconcile(())
            answer = self._assembler.assemble(reconciliation, f"{understanding.task_type.value}_FAILED: {exc}")
            return self._result(execution_id, plan, (), reconciliation, answer, ExecutionStatus.FAILED, trace, request)
        trace = trace.append(TraceStep("tool_execution", ExecutionStatus.EXECUTED, output_references=tuple(item.evidence_id for item in tool_result.evidence), provenance=tool_result.provenance, duration_ms=(perf_counter() - started) * 1000))
        trace = trace.append(TraceStep("evidence_collection", ExecutionStatus.EXECUTED, output_references=tuple(item.evidence_id for item in tool_result.evidence)))
        reconciliation = self._reconciler.reconcile(tool_result.evidence)
        trace = trace.append(TraceStep("reconciliation", reconciliation.status, output_references=tuple(item.evidence_id for item in tool_result.evidence)))
        answer = self._assembler.assemble(reconciliation)
        trace = trace.append(TraceStep("answer_generation", answer.status))
        return self._result(execution_id, plan, (tool_result,), reconciliation, answer, answer.status, trace, request)

    @staticmethod
    def _result(execution_id: str, plan: ExecutionPlan, tool_results: tuple[ToolResult, ...], reconciliation: Reconciliation, answer: AnswerAssembly, status: ExecutionStatus, trace: ExecutionTrace, request: AnalysisRequest) -> AnalysisResult:
        provenance = {"execution_id": execution_id, "query": request.query, "input_ids": [str(item.get("id")) for item in request.inputs], "task_type": plan.understanding.task_type.value, "selected_tool": plan.selected_tool, "representation_references": list(plan.representations)}
        summary = {"status": status.value, "evidence_count": len(reconciliation.evidence), "conflict_count": len(reconciliation.conflicts), "fallback": "NONE"}
        return AnalysisResult(execution_id, plan, tool_results, reconciliation.evidence, answer, status, provenance, summary, trace)
