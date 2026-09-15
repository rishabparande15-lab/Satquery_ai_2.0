"""Executable deterministic scene-analysis capability for Pipeline 3.

This module is an architectural boundary only.  It delegates every scientific
calculation to ``analysis_engine.run_analysis`` and maps that unchanged result
through the Phase 6A.3 contracts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Mapping

from .architecture_contracts import (
    AdaptedAnalysis,
    Capability,
    CapabilityRegistry,
    RepresentationSet,
    SceneBundle,
    TaskRequest,
    TaskResult,
    adapt_run_analysis_output,
)
from .live_representation_persistence import (
    RepresentationPersistencePolicy,
    RepresentationPersistenceSession,
)


CAPABILITY_NAME = "deterministic_scene_analysis"
TASK_TYPE = "deterministic_scene_analysis"
CAPABILITY_VERSION = "1.0.0"
SUPPORTED_MODALITIES = frozenset({"optical", "sar"})


def _rejected(request: TaskRequest, code: str, message: str, *, unavailable: bool = False) -> TaskResult:
    return TaskResult(
        status="unavailable" if unavailable else "failure",
        task_type=request.task_type,
        confidence=None,
        confidence_calibrated=False,
        provenance={"input": {"scene_id": request.scene_id, "scene_reference": request.scene_reference}},
        diagnostics={"code": code, "message": message},
    )


class Pipeline3AnalysisAdapter:
    """Non-mutating call boundary around the authoritative Pipeline 3 runtime."""

    def __init__(self, runner: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None) -> None:
        self._runner = runner

    def execute(
        self,
        request: TaskRequest,
        scene: SceneBundle,
        persistence_policy: RepresentationPersistencePolicy | None = None,
    ) -> AdaptedAnalysis:
        legacy_request = self._legacy_request(request, scene)
        runner = self._runner
        session = RepresentationPersistenceSession(persistence_policy)
        if runner is None:
            from .analysis_engine import run_analysis

            legacy_result = run_analysis(legacy_request, persistence_session=session)
        else:
            if session.policy.enabled:
                raise ValueError("persistence requires the canonical Pipeline 3 runner")
            legacy_result = runner(legacy_request)
        adapted = adapt_run_analysis_output(legacy_result, legacy_request)
        if session.outcome is not None:
            adapted = self._attach_persisted_references(adapted, session)
        return replace(
            adapted,
            task_result=replace(adapted.task_result, task_type=TASK_TYPE),
        )

    @staticmethod
    def _attach_persisted_references(
        adapted: AdaptedAnalysis,
        session: RepresentationPersistenceSession,
    ) -> AdaptedAnalysis:
        outcome = session.outcome
        if outcome is None:
            return adapted
        def identity(reference):
            spatial_level = reference.spatial_reference.level if reference.spatial_reference else None
            return reference.representation_type, reference.modality, spatial_level
        persisted_keys = {identity(reference) for reference in outcome.references}
        existing = tuple(
            reference for reference in adapted.scene.representations.references
            if identity(reference) not in persisted_keys
        )
        representations = replace(
            adapted.scene.representations,
            references=(*existing, *outcome.references),
        )
        persistence = {
            "receipt": outcome.receipt.to_dict(),
            "writes": outcome.writes,
            "reused": outcome.reused,
            "total_bytes": outcome.total_bytes,
        }
        scene_provenance = {**adapted.scene.provenance, "representation_persistence": persistence}
        task_provenance = {**adapted.task_result.provenance, "representation_persistence": persistence}
        artifacts = (
            *adapted.task_result.artifacts,
            outcome.receipt.to_dict(),
            *(reference.artifact.to_dict() for reference in outcome.references if reference.artifact is not None),
        )
        return replace(
            adapted,
            scene=replace(adapted.scene, representations=representations, provenance=scene_provenance),
            task_result=replace(adapted.task_result, provenance=task_provenance, artifacts=artifacts),
        )

    @staticmethod
    def _legacy_request(request: TaskRequest, scene: SceneBundle) -> dict[str, Any]:
        reference = request.scene_reference or {}
        nested = reference.get("request") if isinstance(reference.get("request"), Mapping) else reference
        legacy_request = deepcopy(dict(nested))
        legacy_request.pop("input_type", None)
        legacy_request.pop("request", None)
        if request.scene_id is not None:
            existing = legacy_request.get("sample_id")
            if existing is not None and str(existing) != request.scene_id:
                raise ValueError("scene_id conflicts with scene_reference sample_id")
            legacy_request.setdefault("sample_id", request.scene_id)
        legacy_request["query"] = request.query
        if request.requested_modalities:
            modes = {
                frozenset({"optical"}): "optical_analysis",
                frozenset({"sar"}): "sar_analysis",
                frozenset({"optical", "sar"}): "joint_optical_sar_analysis",
            }
            legacy_request["analysis_type"] = modes[frozenset(request.requested_modalities)]
        for key, value in request.parameters.items():
            legacy_request.setdefault(key, deepcopy(value))
        return legacy_request


class DeterministicSceneAnalysisCapability(Capability):
    """The first executable capability, composed over existing Pipeline 3."""

    def __init__(self, adapter: Pipeline3AnalysisAdapter | None = None) -> None:
        super().__init__(
            name=CAPABILITY_NAME,
            task_type=TASK_TYPE,
            version=CAPABILITY_VERSION,
            supported_modalities=SUPPORTED_MODALITIES,
            supported_input_types=frozenset({"satellite_scene"}),
            supports_temporal=False,
            supports_spatial_evidence=True,
            supports_confidence=False,
        )
        object.__setattr__(self, "_adapter", adapter or Pipeline3AnalysisAdapter())

    def execute(
        self,
        request: TaskRequest,
        scene: SceneBundle | None = None,
        persistence_policy: RepresentationPersistencePolicy | None = None,
    ) -> TaskResult:
        if not isinstance(request, TaskRequest):
            raise TypeError("request must be a TaskRequest")
        if request.task_type != self.task_type:
            return _rejected(request, "unsupported_task_type", f"Expected task type {self.task_type}.")
        if request.temporal_context is not None:
            return _rejected(request, "temporal_not_supported", "Deterministic scene analysis is single-scene only.", unavailable=True)
        requested = frozenset(request.requested_modalities)
        if not requested.issubset(self.supported_modalities):
            return _rejected(request, "unsupported_modalities", "Only optical and SAR scene modalities are supported.", unavailable=True)
        if request.query is None:
            return _rejected(request, "missing_query", "A deterministic analysis query is required.")

        context = scene or self._scene_from_request(request)
        if context.input_type not in self.supported_input_types:
            return _rejected(request, "unsupported_input_type", f"Unsupported input type: {context.input_type}.", unavailable=True)
        if request.scene_id is not None and context.scene_id != request.scene_id:
            return _rejected(request, "scene_mismatch", "TaskRequest and SceneBundle identify different scenes.")
        if context.available_modalities and not requested.issubset(context.available_modalities):
            return _rejected(request, "scene_modalities_unavailable", "The SceneBundle does not provide every requested modality.", unavailable=True)
        if request.scene_id is None and not self._has_legacy_input(request.scene_reference or {}):
            return _rejected(request, "missing_scene_input", "A sample_id or uploaded file reference is required.")

        try:
            return self._adapter.execute(request, context, persistence_policy).task_result
        except (KeyError, TypeError, ValueError) as exc:
            return _rejected(request, "invalid_pipeline3_request", str(exc))

    def execute_with_context(
        self,
        request: TaskRequest,
        scene: SceneBundle | None = None,
        persistence_policy: RepresentationPersistencePolicy | None = None,
    ) -> AdaptedAnalysis:
        """Execute a validated request and retain its populated SceneBundle."""
        if request.task_type != self.task_type or request.query is None:
            raise ValueError("a valid deterministic scene-analysis request is required")
        context = scene or self._scene_from_request(request)
        if not self.supports(request, context):
            raise ValueError("request is not supported by deterministic scene analysis")
        return self._adapter.execute(request, context, persistence_policy)

    @staticmethod
    def _has_legacy_input(reference: Mapping[str, Any]) -> bool:
        nested = reference.get("request") if isinstance(reference.get("request"), Mapping) else reference
        files = nested.get("files")
        return bool(nested.get("sample_id")) or isinstance(files, Mapping) and bool(files)

    @staticmethod
    def _scene_from_request(request: TaskRequest) -> SceneBundle:
        reference = request.scene_reference or {}
        nested = reference.get("request") if isinstance(reference.get("request"), Mapping) else reference
        identifiers = {}
        sample_id = request.scene_id or nested.get("sample_id")
        if sample_id is not None:
            identifiers["sample_id"] = str(sample_id)
        scene_id = str(sample_id or nested.get("scene_id") or "referenced-scene")
        return SceneBundle(
            scene_id=scene_id,
            input_type=str(reference.get("input_type") or "satellite_scene"),
            input_identifiers=identifiers,
            available_modalities=request.requested_modalities,
            representations=RepresentationSet(),
            provenance={"task_request": request.execution_metadata},
        )


def build_capability_registry() -> CapabilityRegistry:
    """Build the production registry containing only the implemented capability."""
    registry = CapabilityRegistry()
    registry.register(DeterministicSceneAnalysisCapability())
    return registry


CAPABILITY_REGISTRY = build_capability_registry()


def run_deterministic_scene_analysis(
    request: TaskRequest,
    scene: SceneBundle | None = None,
    persistence_policy: RepresentationPersistencePolicy | None = None,
) -> TaskResult:
    """Canonical TaskRequest -> registry -> capability -> Pipeline 3 path."""
    capability = CAPABILITY_REGISTRY.require_task_type(request.task_type)
    if not isinstance(capability, DeterministicSceneAnalysisCapability):
        raise TypeError("registered deterministic scene capability is not executable")
    return capability.execute(request, scene, persistence_policy)


def run_deterministic_scene_analysis_with_context(
    request: TaskRequest,
    scene: SceneBundle | None = None,
    persistence_policy: RepresentationPersistencePolicy | None = None,
) -> AdaptedAnalysis:
    """Internal advanced path returning TaskResult plus its representation-bearing scene."""
    capability = CAPABILITY_REGISTRY.require_task_type(request.task_type)
    if not isinstance(capability, DeterministicSceneAnalysisCapability):
        raise TypeError("registered deterministic scene capability is not executable")
    return capability.execute_with_context(request, scene, persistence_policy)
