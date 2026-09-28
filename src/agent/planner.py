"""Explainable keyword planner. It deliberately does not use an LLM."""
from dataclasses import dataclass
from .query_types import AnalysisRequest, QueryUnderstanding, TaskType


@dataclass(frozen=True)
class ExecutionPlan:
    understanding: QueryUnderstanding
    selected_tool: str | None
    representations: tuple[str, ...]
    steps: tuple[str, ...]


def _declared_modality(request: AnalysisRequest) -> str | None:
    values = {str(item.get("modality", "")).lower() for item in request.inputs}
    values.discard("")
    if len(values) == 1:
        return values.pop()
    if values == {"s2", "s1"} or values == {"optical", "sar"}:
        return "optical_sar"
    return None


class DeterministicPlanner:
    def understand(self, request: AnalysisRequest) -> QueryUnderstanding:
        text = request.query.lower()
        modality = _declared_modality(request)
        count = len(request.inputs)
        grounding = any(token in text for token in ("where", "bounding box", "bbox", "mask", "polygon", "point"))
        temporal = count >= 2 or any(token in text for token in ("compare", "change", "before", "after", "temporal"))
        scientific = any(token in text for token in ("land cover", "land-cover", "distribution", "scientific", "classify"))
        rationale: list[str] = []
        if grounding:
            task = TaskType.GROUNDING; rationale.append("grounding keyword")
        elif modality == "optical_sar" or ("optical" in text and "sar" in text):
            task = TaskType.OPTICAL_SAR_REASONING; rationale.append("joint optical/SAR modality")
        elif modality in {"s1", "sar"} or any(token in text for token in ("sar", "radar", "vv/vh")):
            task = TaskType.SAR_VQA; rationale.append("SAR modality")
        elif request.metadata.get("execution_mode") == "temporal_rgb":
            task = TaskType.TEMPORAL_RGB_EXECUTION; rationale.append("explicit validated temporal RGB execution mode")
        elif temporal:
            task = TaskType.CHANGE_VQA if any(token in text for token in ("change", "compare")) else TaskType.TEMPORAL_REASONING; rationale.append("temporal pair/keyword")
        elif scientific and not (modality == "s2" and any(token in text for token in ("describe", "caption"))):
            task = TaskType.SCIENTIFIC_ANALYSIS; rationale.append("scientific-analysis keyword")
        elif modality == "s2" or "sentinel-2" in text or "s2 " in text:
            task = TaskType.S2_CAPTIONING if any(token in text for token in ("describe", "caption")) else TaskType.S2_VQA; rationale.append("S2 modality")
        elif "report" in text:
            task = TaskType.REPORT_GENERATION; rationale.append("report keyword")
        else:
            task = TaskType.RGB_VISUAL_REASONING; rationale.append("default RGB visual request")
        output = "scientific_prediction" if task == TaskType.SCIENTIFIC_ANALYSIS else ("grounding" if grounding else "language")
        return QueryUnderstanding(request.query, task, modality, count, temporal, grounding, scientific, output, (task,), rationale=tuple(rationale))

    def plan(self, request: AnalysisRequest, tool_name: str | None, representations: tuple[str, ...]) -> ExecutionPlan:
        understood = self.understand(request)
        return ExecutionPlan(understood, tool_name, representations, ("query_interpretation", "validation", "representation_selection", "tool_selection", "tool_execution", "evidence_collection", "reconciliation", "answer_generation"))
