"""Unified, fail-closed orchestration over SatQuery's existing specialists.

This module owns routing, response normalization, operational traces, and
capability disclosure.  It deliberately does not load, alter, or train models.
Specialist execution is injected so each validated controller remains the sole
owner of its model-specific input contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, Callable, Mapping
from uuid import uuid4


AVAILABLE_WITH_LIMITATIONS = "AVAILABLE_WITH_LIMITATIONS"
AVAILABLE = "AVAILABLE"
BLOCKED = "BLOCKED"
EXPERIMENTAL_LIMITED = "EXPERIMENTAL_LIMITED"


CAPABILITIES: dict[str, dict[str, Any]] = {
    "SINGLE_IMAGE_VQA": {
        "display_name": "Single-image VQA", "status": AVAILABLE_WITH_LIMITATIONS,
        "required_inputs": ["one_s2_optical_image"], "supported_query_types": ["VQA"],
        "specialist": "phase3o5_s2_projector_qwen25vl",
        "model_version": "S2 projector e7f22d74… + Qwen 66285546…",
        "output_type": "answer", "availability": "AVAILABLE",
        "limitations": ["Internal validation only; benchmark generalization is not established."],
    },
    "SINGLE_IMAGE_SAR_VQA": {
        "display_name": "Single-image SAR VQA", "status": AVAILABLE_WITH_LIMITATIONS,
        "required_inputs": ["one_s1_sar_image"], "supported_query_types": ["VQA"],
        "specialist": "official_croma_s1 + phase3p1_s1_sar_projector + qwen25vl",
        "model_version": "S1 projector a7e57bff… + Qwen 66285546…",
        "output_type": "answer", "availability": "AVAILABLE",
        "limitations": ["SAR_ONLY_SEMANTIC_VALIDATION_LIMITED: the route is technically conditioned on SAR, but standalone SAR did not demonstrate clear semantic improvement."],
    },
    "OPTICAL_SAR_ANALYSIS": {
        "display_name": "Optical + SAR analysis", "status": AVAILABLE_WITH_LIMITATIONS,
        "required_inputs": ["one_s1_image", "one_s2_image"], "supported_query_types": ["OPTICAL_SAR_REASONING"],
        "specialist": "CROMA joint representation + CromaJointProjector + Qwen2.5-VL-3B-Instruct",
        "model_version": "joint projector bedcb8dc… + Qwen 66285546…",
        "output_type": "answer", "availability": "AVAILABLE",
        "limitations": ["Multimodal path is technically valid but did not demonstrate improvement over the optical-only baseline."],
    },
    "TEMPORAL_CHANGE_DESCRIPTION": {
        "display_name": "Bi-temporal change description", "status": AVAILABLE,
        "required_inputs": ["one_t1_pre_image", "one_t2_post_image"], "supported_query_types": ["CHANGE_DESCRIPTION"],
        "specialist": "Chg2Cap", "model_version": "LEVIR_CC_batchsize_32_resnet101.pth d737a92a…",
        "output_type": "change_description", "availability": "AVAILABLE",
        "limitations": ["Natural-language change description only; no spatial change mask or area estimate is produced."],
    },
    "SINGLE_IMAGE_GROUNDING": {
        "display_name": "Text-guided grounding", "status": BLOCKED,
        "required_inputs": ["one_image"], "supported_query_types": ["GROUNDING"], "specialist": None,
        "model_version": None, "output_type": "none", "availability": "BLOCKED",
        "limitations": ["No validated grounding specialist or trustworthy pixel-coordinate mapping is integrated."],
    },
    "SINGLE_IMAGE_CAPTION": {
        "display_name": "Single-image captioning", "status": EXPERIMENTAL_LIMITED,
        "required_inputs": ["one_s2_optical_image"], "supported_query_types": ["CAPTION_REQUEST"], "specialist": None,
        "model_version": None, "output_type": "unreliable_caption", "availability": "EXPERIMENTAL_LIMITED",
        "limitations": ["Single-image captioning is experimental and currently unreliable."],
    },
    "SINGLE_IMAGE_SCENE_DESCRIPTION": {
        "display_name": "Single-image scene description", "status": AVAILABLE,
        "required_inputs": ["one_explicit_rgb_image_or_s2_rgb_rendering"], "supported_query_types": ["SCENE_DESCRIPTION"],
        "specialist": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct",
        "model_version": "model.safetensors 7901956b…", "output_type": "scene_description", "availability": "AVAILABLE",
        "limitations": ["Scene description is generated from an explicit RGB rendering or RGB upload; sensor identity is not inferred from pixels.", "Answer confidence is not calibrated."],
    },
}


class AgentInputError(ValueError):
    """A user-safe routing or input configuration rejection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InputConfiguration:
    kind: str
    normalized_inputs: tuple[dict[str, Any], ...]
    basis: str


def capability_registry() -> dict[str, dict[str, Any]]:
    """Return copy-safe registry metadata for API/UI disclosure."""
    return {name: {key: list(value) if isinstance(value, tuple) else value for key, value in item.items()} for name, item in CAPABILITIES.items()}


def classify_input_configuration(inputs: list[Mapping[str, Any]]) -> InputConfiguration:
    if not isinstance(inputs, list) or not inputs:
        raise AgentInputError("INPUTS_REQUIRED", "At least one declared input is required.")
    normal: list[dict[str, Any]] = []
    roles: set[str] = set()
    modalities: set[str] = set()
    for index, raw in enumerate(inputs):
        if not isinstance(raw, Mapping):
            raise AgentInputError("INVALID_INPUT", "Each input must be an object.")
        role = str(raw.get("role", "")).upper()
        modality = str(raw.get("modality", "")).lower()
        input_id = str(raw.get("input_id") or raw.get("id") or role or f"input_{index + 1}")
        if role not in {"SINGLE", "S1", "S2", "T1", "T2"}:
            raise AgentInputError("INVALID_INPUT_ROLE", "Each input role must be SINGLE, S1, S2, T1, or T2.")
        if role in roles:
            raise AgentInputError("DUPLICATE_INPUT_ROLE", f"Duplicate {role} input is not allowed.")
        if not modality:
            raise AgentInputError("INPUT_MODALITY_REQUIRED", "Each input requires an explicit modality.")
        roles.add(role); modalities.add(modality)
        normal.append({"input_id": input_id, "role": role, "modality": modality,
                       "source": raw.get("source"), "file_type": raw.get("file_type"),
                       "sensor": raw.get("sensor"), "shape": raw.get("shape"),
                       "channels": raw.get("channels"), "crs": raw.get("crs"),
                       "resolution": raw.get("resolution"), "bounds": raw.get("bounds"),
                       "pair_id": raw.get("pair_id"), "temporal_role": raw.get("temporal_role"),
                       "split": raw.get("split"), "provenance": raw.get("provenance"),
                       "external_descriptor": raw.get("external_descriptor"),
                       "adapter": raw.get("adapter"), "compatibility": raw.get("compatibility")})
    if roles == {"T1", "T2"}:
        return InputConfiguration("TEMPORAL_PAIR", tuple(normal), "Two inputs are explicitly labelled T1 and T2.")
    if roles & {"T1", "T2"}:
        raise AgentInputError("INCOMPLETE_TEMPORAL_PAIR", "A temporal request requires both explicitly ordered T1 / PRE and T2 / POST inputs.")
    if roles == {"S1", "S2"}:
        return InputConfiguration("CROSS_MODAL_PAIR", tuple(normal), "Inputs are explicitly labelled S1 and S2.")
    if roles & {"S1", "S2"}:
        raise AgentInputError("INCOMPLETE_OPTICAL_SAR_PAIR", "Optical + SAR analysis requires both S1 and S2 inputs.")
    if roles == {"SINGLE"}:
        return InputConfiguration("SINGLE_IMAGE_CANDIDATE", tuple(normal), "One image is explicitly declared as SINGLE.")
    raise AgentInputError("UNSUPPORTED_INPUT_CONFIGURATION", "Input roles do not form a supported SatQuery configuration.")


def classify_query_intent(query: str) -> tuple[str, str]:
    text = str(query or "").strip().lower()
    if not text:
        raise AgentInputError("QUERY_REQUIRED", "A non-empty query is required.")
    if any(word in text for word in ("highlight", "locate", "outline", "bounding box", "bbox", "mask", "polygon", "point")):
        return "GROUNDING", "The query requests a spatial location or overlay."
    if any(word in text for word in ("what changed", "changes between", "before and after", "pre-change", "post-change", "change description")):
        return "CHANGE_DESCRIPTION", "The query explicitly requests change between dates."
    if "optical and sar" in text:
        return "OPTICAL_SAR_REASONING", "The query explicitly requests optical/SAR reasoning."
    if any(word in text for word in ("describe this image", "caption this image", "caption the image", "describe the scene", "scene description")):
        return "SCENE_DESCRIPTION", "The query explicitly requests a single-image scene description."
    return "VQA", "The query is a single-image visual question."


def select_route(configuration: InputConfiguration, intent: str, requested_task: str | None = None) -> tuple[str, str]:
    requested = str(requested_task or "").strip().upper() or None
    if requested and requested not in CAPABILITIES:
        raise AgentInputError("UNSUPPORTED_REQUESTED_TASK", "The requested task is not a registered SatQuery capability.")
    single = configuration.normalized_inputs[0] if configuration.kind == "SINGLE_IMAGE_CANDIDATE" else {}
    is_s1_sar = str(single.get("sensor") or "").strip().lower() in {"sentinel-1", "s1"} and str(single.get("modality") or "").lower() == "sar"
    expected = {
        ("SINGLE_IMAGE_CANDIDATE", "VQA"): "SINGLE_IMAGE_SAR_VQA" if is_s1_sar else "SINGLE_IMAGE_VQA",
        ("SINGLE_IMAGE_CANDIDATE", "GROUNDING"): "SINGLE_IMAGE_GROUNDING",
        ("SINGLE_IMAGE_CANDIDATE", "SCENE_DESCRIPTION"): "SINGLE_IMAGE_SCENE_DESCRIPTION",
        ("CROSS_MODAL_PAIR", "OPTICAL_SAR_REASONING"): "OPTICAL_SAR_ANALYSIS",
        ("CROSS_MODAL_PAIR", "VQA"): "OPTICAL_SAR_ANALYSIS",
        ("TEMPORAL_PAIR", "CHANGE_DESCRIPTION"): "TEMPORAL_CHANGE_DESCRIPTION",
    }.get((configuration.kind, intent))
    if requested:
        if requested != expected:
            raise AgentInputError("REQUESTED_TASK_INCOMPATIBLE", "The selected task is incompatible with the declared input configuration and query intent.")
        return requested, f"User-selected {requested} is compatible with declared inputs and query intent."
    if expected:
        return expected, f"{configuration.basis} {intent}: selected {expected}."
    if configuration.kind == "TEMPORAL_PAIR":
        raise AgentInputError("ROUTING_CLARIFICATION_REQUIRED", "PRE/POST inputs require a change-description query; SatQuery will not silently reinterpret them as single-image VQA.")
    if configuration.kind == "CROSS_MODAL_PAIR":
        raise AgentInputError("ROUTING_CLARIFICATION_REQUIRED", "S1+S2 inputs require an optical-SAR query; SatQuery will not discard S1 or silently route to single-image VQA.")
    raise AgentInputError("ROUTING_CLARIFICATION_REQUIRED", "The query and input configuration do not identify a safe specialist.")


class SatQueryAgent:
    """Central dispatcher whose injected executors call existing specialists."""

    def __init__(self, executors: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]) -> None:
        self._executors = dict(executors)

    def run(self, *, inputs: list[Mapping[str, Any]], query: str, context: Mapping[str, Any] | None = None,
            requested_task: str | None = None) -> dict[str, Any]:
        request_id = str(uuid4()); started = perf_counter(); context = dict(context or {})
        trace: list[dict[str, Any]] = []
        def step(component: str, action: str, status: str, detail: str, duration_ms: float | None = None) -> None:
            item = {"step_id": len(trace) + 1, "component": component, "action": action, "status": status,
                    "detail": detail, "timestamp": time()}
            if duration_ms is not None: item["duration_ms"] = round(duration_ms, 3)
            trace.append(item)
        try:
            config = classify_input_configuration(inputs)
            step("INPUT_CONFIGURATION", "classify", "COMPLETED", config.basis)
            intent, intent_basis = classify_query_intent(query)
            step("QUERY_CLASSIFICATION", "classify", "COMPLETED", intent_basis)
            route, routing_basis = select_route(config, intent, requested_task)
            step("ROUTE_SELECTION", "select_capability", "COMPLETED", routing_basis)
        except AgentInputError as exc:
            step("ROUTE_SELECTION", "reject", "BLOCKED", str(exc))
            return self._error(request_id, query, exc.code, str(exc), trace)
        capability = CAPABILITIES[route]
        if capability["status"] == BLOCKED:
            step("CAPABILITY_POLICY", "availability_check", "BLOCKED", capability["limitations"][0])
            return self._blocked(request_id, query, route, capability, config, trace)
        if capability["status"] == EXPERIMENTAL_LIMITED:
            step("CAPABILITY_POLICY", "availability_check", "BLOCKED", capability["limitations"][0])
            return self._blocked(request_id, query, route, capability, config, trace, "EXPERIMENTAL_CAPTION_UNAVAILABLE")
        executor = self._executors.get(route)
        if executor is None:
            step("SPECIALIST_EXECUTION", "resolve_executor", "FAILED", "No configured specialist executor.")
            return self._error(request_id, query, "SPECIALIST_UNAVAILABLE", "The selected specialist is not available in this runtime.", trace, route)
        step("INPUT_VALIDATION", "validate", "COMPLETED", "Declared inputs passed configuration-level validation.")
        began = perf_counter()
        try:
            specialist = dict(executor({"inputs": config.normalized_inputs, "query": query, "context": context, "route": route}))
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            step("SPECIALIST_EXECUTION", "run", "FAILED", str(exc), (perf_counter() - began) * 1000)
            return self._error(request_id, query, "SPECIALIST_EXECUTION_FAILED", str(exc), trace, route)
        step("SPECIALIST_EXECUTION", "run", "COMPLETED", f"{capability['specialist']} completed.", (perf_counter() - began) * 1000)
        answer = specialist.get("change_description") if route == "TEMPORAL_CHANGE_DESCRIPTION" else specialist.get("answer") or specialist.get("generated_response")
        if not isinstance(answer, str) or not answer.strip():
            step("RESULT_INTEGRATION", "normalize", "FAILED", "Specialist returned no non-empty language output.")
            return self._error(request_id, query, "EMPTY_SPECIALIST_OUTPUT", "The specialist did not return a non-empty result.", trace, route)
        step("RESULT_INTEGRATION", "normalize", "COMPLETED", "A common response contract was assembled.")
        provenance = {"route": route, "input_summary": list(config.normalized_inputs), "specialist": capability["specialist"],
                      "specialist_provenance": specialist.get("provenance"), "source_metadata": specialist.get("source_metadata"),
                      "model": specialist.get("selected_specialist") or specialist.get("model_tool") or capability["specialist"]}
        return {"status": "COMPLETED", "request_id": request_id, "query": query, "route": route,
                "capability_status": capability["status"], "answer": answer.strip(), "visual_evidence": self._evidence(route, specialist),
                "confidence": {"value": None, "type": "NOT_AVAILABLE"},
                "warnings": list(dict.fromkeys([*capability["limitations"], *list(specialist.get("warnings") or [])])),
                "provenance": provenance, "execution_trace": trace,
                "details": specialist, "error": None, "runtime_seconds": round(perf_counter() - started, 6),
                "routing_basis": routing_basis}

    @staticmethod
    def _evidence(route: str, specialist: Mapping[str, Any]) -> dict[str, Any]:
        if route == "TEMPORAL_CHANGE_DESCRIPTION":
            return {"type": "INPUT_PAIR_PREVIEWS", "t1": specialist.get("t1_identity"), "t2": specialist.get("t2_identity"), "masks": [], "boxes": []}
        if route in {"SINGLE_IMAGE_VQA", "SINGLE_IMAGE_SAR_VQA", "SINGLE_IMAGE_SCENE_DESCRIPTION"}: return specialist.get("visual_evidence") or {"type": "INPUT_IMAGE_ONLY", "masks": [], "boxes": []}
        return {"type": "INPUT_MODALITY_PAIR", "modalities": specialist.get("modalities_used", ["S1", "S2"]), "masks": [], "boxes": []}

    @staticmethod
    def _error(request_id: str, query: str, code: str, message: str, trace: list[dict[str, Any]], route: str | None = None) -> dict[str, Any]:
        return {"status": "ERROR", "request_id": request_id, "query": query, "route": route, "capability_status": "UNSUPPORTED",
                "answer": None, "visual_evidence": None, "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings": [],
                "provenance": {}, "execution_trace": trace, "details": {}, "error": {"code": code, "message": message}}

    @staticmethod
    def _blocked(request_id: str, query: str, route: str, capability: Mapping[str, Any], configuration: InputConfiguration,
                 trace: list[dict[str, Any]], code: str = "GROUNDING_MODEL_UNAVAILABLE") -> dict[str, Any]:
        return {"status": "BLOCKED", "request_id": request_id, "query": query, "route": route,
                "capability_status": capability["status"], "answer": None, "visual_evidence": {"type": "NONE", "masks": [], "boxes": []},
                "confidence": {"value": None, "type": "NOT_AVAILABLE"}, "warnings": capability["limitations"],
                "provenance": {"route": route, "input_summary": list(configuration.normalized_inputs), "specialist": None},
                "execution_trace": trace, "details": {}, "error": {"code": code, "message": capability["limitations"][0]}}
