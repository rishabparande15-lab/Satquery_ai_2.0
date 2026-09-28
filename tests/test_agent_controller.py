from src.agent.capabilities import CapabilityState, default_capability_registry
from src.agent.controller import AgentController, Tool, ToolResult
from src.agent.evidence import ConfidenceSource, Evidence, EvidenceType
from src.agent.query_types import AnalysisRequest, ExecutionStatus, TaskType
from src.agent.reconciler import EvidenceReconciler


class FakeTool(Tool):
    def __init__(self, name, evidence): self.name, self.evidence, self.calls = name, evidence, 0
    def execute(self, request, *, representations):
        self.calls += 1
        return ToolResult({"representations": representations}, self.evidence, {"model_id": self.name})


def request(query, modality="rgb", count=1):
    return AnalysisRequest(query, tuple({"id": f"image-{index}", "modality": modality} for index in range(count)))


def evidence(kind, claim="forest", key="cover"):
    return Evidence("e-1", kind, "fake", claim, key, confidence_source=ConfidenceSource.UNKNOWN, provenance={"model_id": "fake"})


def test_registry_preserves_current_boundaries():
    registry = default_capability_registry()
    assert registry.get(TaskType.SCIENTIFIC_ANALYSIS).state == CapabilityState.IMPLEMENTED
    assert registry.get(TaskType.S2_VQA).state == CapabilityState.EXPERIMENTAL
    assert registry.get(TaskType.SAR_VQA).state == CapabilityState.BLOCKED


def test_scientific_route_keeps_hybrid_scientific_only():
    tool = FakeTool("scientific_predictor", (evidence(EvidenceType.SCIENTIFIC_PREDICTION),))
    result = AgentController({tool.name: tool}).analyze(request("What land-cover distribution is present?", "s2"))
    assert result.plan.understanding.task_type == TaskType.SCIENTIFIC_ANALYSIS
    assert "hybrid_830d" in result.plan.representations
    assert result.trace.steps[2].provenance["scientific_predictor_only"] is True
    assert tool.calls == 1


def test_rgb_and_s2_are_routed_to_injected_tools_only():
    rgb = FakeTool("qwen_rgb", (evidence(EvidenceType.MODEL_LANGUAGE_OUTPUT),))
    s2 = FakeTool("qwen_s2", (evidence(EvidenceType.MODEL_LANGUAGE_OUTPUT),))
    controller = AgentController({rgb.name: rgb, s2.name: s2})
    assert controller.analyze(request("Describe this image.")).plan.understanding.task_type == TaskType.RGB_VISUAL_REASONING
    assert controller.analyze(request("Describe this Sentinel-2 image.", "s2")).plan.understanding.task_type == TaskType.S2_CAPTIONING
    assert rgb.calls == 1 and s2.calls == 1


def test_blocked_requests_do_not_invoke_a_fallback_tool():
    fake = FakeTool("qwen_rgb", (evidence(EvidenceType.MODEL_LANGUAGE_OUTPUT),))
    controller = AgentController({fake.name: fake})
    for query, modality, count, expected in (("What objects are in this SAR image?", "s1", 1, TaskType.SAR_VQA), ("Use optical and SAR together", "s2", 1, TaskType.OPTICAL_SAR_REASONING), ("Compare these images", "rgb", 2, TaskType.CHANGE_VQA), ("Where is the forest? Give a mask", "rgb", 1, TaskType.GROUNDING)):
        result = controller.analyze(request(query, modality, count))
        assert result.plan.understanding.task_type == expected
        assert result.capability_status == ExecutionStatus.BLOCKED
        assert "Fallback: NONE" in result.final_answer.text
        assert [step.name for step in result.trace.steps] == ["query_interpretation", "validation", "representation_selection", "tool_selection", "tool_execution", "evidence_collection", "reconciliation", "answer_generation"]
    assert fake.calls == 0


def test_language_evidence_never_becomes_scientific_and_conflicts_preserved():
    language = evidence(EvidenceType.MODEL_LANGUAGE_OUTPUT, "water")
    assert language.evidence_type == EvidenceType.MODEL_LANGUAGE_OUTPUT
    scientific = Evidence("e-2", EvidenceType.SCIENTIFIC_PREDICTION, "probe", "forest", "cover", confidence=0.65, confidence_source=ConfidenceSource.SCIENTIFIC_METRIC)
    reconciliation = EvidenceReconciler().reconcile((language, scientific))
    assert reconciliation.status == ExecutionStatus.CONFLICT
    assert reconciliation.conflicts[0].claims == ("water", "forest")


def test_malformed_request_and_missing_tool_fail_closed():
    controller = AgentController()
    try:
        controller.analyze(AnalysisRequest("Describe", ()))
    except ValueError as exc:
        assert "declared input" in str(exc)
    result = controller.analyze(request("Describe this image."))
    assert result.capability_status == ExecutionStatus.NOT_VERIFIED


def test_planning_is_deterministic():
    controller = AgentController()
    first = controller._planner.understand(request("What land-cover distribution is present?", "s2"))
    second = controller._planner.understand(request("What land-cover distribution is present?", "s2"))
    assert first == second
