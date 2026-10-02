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
        image_task_mode = str(request.metadata.get("image_task_mode", "")).upper()
        grounding = any(token in text for token in ("highlight", "locate", "outline", "where", "bounding box", "bbox", "mask", "polygon", "point"))
        temporal_intent = any(token in text for token in ("compare", "change", "before", "after", "temporal"))
        temporal = count >= 2 or temporal_intent
        scientific = any(token in text for token in ("land cover", "land-cover", "distribution", "scientific", "classify"))
        rationale: list[str] = []
        if image_task_mode == "BI_TEMPORAL":
            if modality == "optical_sar":
                task = TaskType.OPTICAL_SAR_ANALYSIS; rationale.append("optical/SAR pair is not a temporal pair")
            elif count != 2:
                task = TaskType.TEMPORAL_ROUTE_NOT_IMPLEMENTED; rationale.append("bi-temporal route requires exactly two ordered images")
            else:
                task = TaskType.TEMPORAL_CHANGE_DESCRIPTION; rationale.append("explicit bi-temporal PRE/POST mode")
        elif image_task_mode == "SINGLE_IMAGE" and temporal_intent:
            task = TaskType.TEMPORAL_ROUTE_NOT_IMPLEMENTED; rationale.append("temporal route disabled")
        elif image_task_mode == "SINGLE_IMAGE" and grounding:
            task = TaskType.SINGLE_IMAGE_GROUNDING; rationale.append("single-image grounding intent")
        elif image_task_mode == "SINGLE_IMAGE" and modality == "optical_sar":
            task = TaskType.OPTICAL_SAR_ANALYSIS; rationale.append("explicit paired optical/SAR input")
        elif image_task_mode == "SINGLE_IMAGE" and count > 1:
            task = TaskType.TEMPORAL_ROUTE_NOT_IMPLEMENTED; rationale.append("multi-image route disabled")
        elif image_task_mode == "SINGLE_IMAGE":
            task = TaskType.SINGLE_IMAGE_VQA; rationale.append("explicit single-image VQA mode")
        elif grounding:
            task = TaskType.GROUNDING; rationale.append("grounding keyword")
        elif modality == "optical_sar" or ("optical" in text and "sar" in text):
            task = TaskType.OPTICAL_SAR_REASONING; rationale.append("joint optical/SAR modality")
        elif modality in {"s1", "sar"} or any(token in text for token in ("sar", "radar", "vv/vh")):
            task = TaskType.SAR_VQA; rationale.append("SAR modality")
        elif request.metadata.get("execution_mode") == "temporal_rgb":
            task = TaskType.TEMPORAL_RGB_EXECUTION; rationale.append("explicit validated temporal RGB execution mode")
        elif temporal:
            task = TaskType.TEMPORAL_CHANGE_DESCRIPTION; rationale.append("temporal pair/keyword")
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
