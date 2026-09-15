"""Lightweight contracts around the canonical Pipeline 3 analysis runtime.

These types describe data crossing capability boundaries.  They deliberately do
not execute models, alter preprocessing, or replace the established spatial
evidence schema.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
from typing import Any, Mapping


CONTRACT_VERSION = "architecture_contracts_v1"
RESULT_STATUSES = frozenset({"success", "partial", "failure", "unavailable"})
_KNOWN_MODALITY_REQUIREMENTS = {
    "optical_sar_analysis": frozenset({"optical", "sar"}),
}
_QUERY_REQUIRED_TASKS = frozenset({"vqa"})
_TEMPORAL_TASKS = frozenset({"temporal_change"})


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _strings(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a sequence of strings")
    result = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(item.strip().lower() for item in result)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


class RepresentationType(str, Enum):
    RAW_OPTICAL = "raw_optical"
    RAW_SAR = "raw_sar"
    PHYSICAL_FEATURES = "physical_features"
    OPTICAL_CROMA = "optical_croma"
    SAR_CROMA = "sar_croma"
    JOINT_CROMA = "joint_croma"
    CROMA_SCENE = "croma_scene"
    HYBRID = "hybrid"
    REGIONS = "regions"
    EVIDENCE = "evidence"
    METADATA = "metadata"


class ArtifactType(str, Enum):
    NUMPY_ARRAY = "numpy_array"
    JSON_DOCUMENT = "json_document"
    IMAGE = "image"
    ANALYSIS_REPORT = "analysis_report"
    PROVENANCE_RECEIPT = "provenance_receipt"


_SPATIAL_LEVELS = frozenset({"pixel", "feature", "scene", "token", "region", "evidence", "metadata"})
_REPRESENTATION_STATUSES = frozenset({"available", "materialized", "not_addressable", "unavailable"})
_ARTIFACT_STATUSES = frozenset({"available", "unavailable"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DTYPE = re.compile(r"^(?:bool|(?:u?int|float)(?:8|16|32|64)|json)$")
_LOGICAL_URI = re.compile(r"^artifact://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+$")


@dataclass(frozen=True)
class SpatialReference:
    """Truthful spatial level and optional mapping metadata."""

    level: str
    crs: str | None = None
    token_grid: tuple[int, int] | None = None
    mapping_reference: str | None = None
    mapping_version: str | None = None

    def __post_init__(self) -> None:
        level = _nonempty(self.level, "level").lower()
        if level not in _SPATIAL_LEVELS:
            raise ValueError(f"level must be one of {sorted(_SPATIAL_LEVELS)}")
        object.__setattr__(self, "level", level)
        if self.crs is not None:
            _nonempty(self.crs, "crs")
        if self.token_grid is not None and (
            len(self.token_grid) != 2 or any(type(value) is not int or value <= 0 for value in self.token_grid)
        ):
            raise ValueError("token_grid must contain two positive integers")
        if self.mapping_reference is not None:
            _nonempty(self.mapping_reference, "mapping_reference")
        if self.mapping_version is not None:
            _nonempty(self.mapping_version, "mapping_version")


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable identity for a concrete artifact at a logical location."""

    artifact_id: str
    artifact_type: ArtifactType
    uri: str
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    format: str | None = None
    producer: str | None = None
    producer_version: str | None = None
    run_reference: str | None = None
    provenance_reference: str | None = None
    status: str = "available"

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        try:
            object.__setattr__(self, "artifact_type", ArtifactType(self.artifact_type))
        except ValueError as exc:
            raise ValueError("invalid artifact_type") from exc
        uri = _nonempty(self.uri, "uri")
        if not _LOGICAL_URI.fullmatch(uri) or ".." in uri.split("/"):
            raise ValueError("uri must be a stable artifact:// logical reference")
        if self.checksum_sha256 is not None:
            checksum = self.checksum_sha256.lower()
            if not _SHA256.fullmatch(checksum):
                raise ValueError("checksum_sha256 must contain 64 hexadecimal characters")
            object.__setattr__(self, "checksum_sha256", checksum)
        if self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes < 0):
            raise ValueError("size_bytes must be a non-negative integer")
        for name in ("media_type", "format", "producer", "producer_version", "run_reference", "provenance_reference"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        if self.status not in _ARTIFACT_STATUSES:
            raise ValueError(f"status must be one of {sorted(_ARTIFACT_STATUSES)}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifact_type"] = self.artifact_type.value
        return result


@dataclass(frozen=True)
class RepresentationRef:
    """Immutable metadata identity, separate from loaded representation material."""

    reference_id: str
    scene_id: str
    representation_type: RepresentationType
    modality: str | None = None
    version: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    spatial_reference: SpatialReference | None = None
    producer: str | None = None
    producer_version: str | None = None
    preprocessing_version: str | None = None
    artifact: ArtifactRef | None = None
    provenance_reference: str | None = None
    status: str = "unavailable"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _nonempty(self.reference_id, "reference_id"))
        object.__setattr__(self, "scene_id", _nonempty(self.scene_id, "scene_id"))
        try:
            object.__setattr__(self, "representation_type", RepresentationType(self.representation_type))
        except ValueError as exc:
            raise ValueError("invalid representation_type") from exc
        if self.modality is not None:
            object.__setattr__(self, "modality", _nonempty(self.modality, "modality").lower())
        for name in ("version", "producer", "producer_version", "preprocessing_version", "provenance_reference"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        if self.shape is not None and (
            not self.shape or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise ValueError("shape must contain positive integers")
        if self.dtype is not None:
            dtype = _nonempty(self.dtype, "dtype").lower()
            if not _DTYPE.fullmatch(dtype):
                raise ValueError("dtype must be a supported numeric, bool, or json type")
            object.__setattr__(self, "dtype", dtype)
        if self.spatial_reference is not None and not isinstance(self.spatial_reference, SpatialReference):
            raise ValueError("spatial_reference must be a SpatialReference")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactRef):
            raise ValueError("artifact must be an ArtifactRef")
        if self.status not in _REPRESENTATION_STATUSES:
            raise ValueError(f"status must be one of {sorted(_REPRESENTATION_STATUSES)}")
        if self.status == "available" and self.artifact is None:
            raise ValueError("available representations require an artifact reference")
        if self.artifact is not None and self.artifact.status != "available" and self.status == "available":
            raise ValueError("available representations require an available artifact")

    @property
    def addressable(self) -> bool:
        return self.status == "available" and self.artifact is not None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["representation_type"] = self.representation_type.value
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_dict()
        return result


@dataclass(frozen=True)
class RepresentationSet:
    """Optional current and future representations for one scene."""

    physical_features: Any = None
    optical: Any = None
    sar: Any = None
    joint: Any = None
    croma_scene: Any = None
    spatial_tokens: Any = None
    regions: Any = None
    references: tuple[RepresentationRef, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(reference, RepresentationRef) for reference in self.references):
            raise ValueError("references must contain RepresentationRef values")
        keys = tuple(reference.reference_id for reference in self.references)
        if len(set(keys)) != len(keys):
            raise ValueError("representation reference_id values must be unique")

    def find(
        self,
        representation_type: RepresentationType | str,
        *,
        modality: str | None = None,
        spatial_level: str | None = None,
        addressable_only: bool = False,
    ) -> tuple[RepresentationRef, ...]:
        kind = RepresentationType(representation_type)
        normalized_modality = modality.lower() if modality is not None else None
        return tuple(
            reference for reference in self.references
            if reference.representation_type == kind
            and (normalized_modality is None or reference.modality == normalized_modality)
            and (spatial_level is None or (
                reference.spatial_reference is not None
                and reference.spatial_reference.level == spatial_level.lower()
            ))
            and (not addressable_only or reference.addressable)
        )

    def with_reference(self, reference: RepresentationRef) -> "RepresentationSet":
        """Return a new set; the existing immutable set is never modified."""
        if not isinstance(reference, RepresentationRef):
            raise TypeError("reference must be a RepresentationRef")
        return RepresentationSet(
            physical_features=self.physical_features,
            optical=self.optical,
            sar=self.sar,
            joint=self.joint,
            croma_scene=self.croma_scene,
            spatial_tokens=self.spatial_tokens,
            regions=self.regions,
            references=(*self.references, reference),
        )


@dataclass(frozen=True)
class SceneBundle:
    """Canonical reusable context for a Pipeline 3 scene.

    Optional fields are intentional: a valid optical-only or SAR-only scene does
    not need temporal, grounding, joint, or CROMA artifacts.
    """

    scene_id: str
    input_type: str = "satellite_scene"
    input_identifiers: Mapping[str, str] = field(default_factory=dict)
    analysis_id: str | None = None
    available_modalities: tuple[str, ...] = ()
    optical_available: bool | None = None
    sar_available: bool | None = None
    temporal_available: bool | None = None
    temporal_context: Mapping[str, Any] | None = None
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    resolution: tuple[float, float] | None = None
    dimensions: tuple[int, int] | None = None
    georeferencing_status: str = "unknown"
    coregistration_status: str = "not_applicable"
    representations: RepresentationSet = field(default_factory=RepresentationSet)
    spatial_evidence: Mapping[str, Any] | None = None
    task_evidence_references: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _nonempty(self.scene_id, "scene_id"))
        object.__setattr__(self, "input_type", _nonempty(self.input_type, "input_type"))
        if self.analysis_id is not None:
            _nonempty(self.analysis_id, "analysis_id")
        _mapping(self.input_identifiers, "input_identifiers")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in self.input_identifiers.items()):
            raise ValueError("input_identifiers must map strings to strings")
        modalities = _strings(self.available_modalities, "available_modalities")
        object.__setattr__(self, "available_modalities", modalities)
        for name in ("optical", "sar", "temporal"):
            attribute = f"{name}_available"
            supplied = getattr(self, attribute)
            inferred = name in modalities
            if supplied is None:
                object.__setattr__(self, attribute, inferred)
            elif type(supplied) is not bool or supplied != inferred:
                raise ValueError(f"{attribute} must match available_modalities")
        if self.temporal_context is not None:
            _mapping(self.temporal_context, "temporal_context")
        if self.crs is not None:
            _nonempty(self.crs, "crs")
        if self.bounds is not None:
            if len(self.bounds) != 4 or not all(math.isfinite(float(v)) for v in self.bounds):
                raise ValueError("bounds must contain four finite numbers")
            if self.bounds[0] >= self.bounds[2] or self.bounds[1] >= self.bounds[3]:
                raise ValueError("bounds must be ordered left, bottom, right, top")
        if self.resolution is not None and (len(self.resolution) != 2 or any(float(v) <= 0 or not math.isfinite(float(v)) for v in self.resolution)):
            raise ValueError("resolution must contain two positive finite numbers")
        if self.dimensions is not None and (len(self.dimensions) != 2 or any(type(v) is not int or v <= 0 for v in self.dimensions)):
            raise ValueError("dimensions must contain two positive integers")
        if self.georeferencing_status not in {"unknown", "available", "validated", "unavailable"}:
            raise ValueError("invalid georeferencing_status")
        if self.coregistration_status not in {"not_applicable", "unknown", "aligned", "misaligned", "unavailable"}:
            raise ValueError("invalid coregistration_status")
        if not isinstance(self.representations, RepresentationSet):
            raise ValueError("representations must be a RepresentationSet")
        if self.spatial_evidence is not None:
            _mapping(self.spatial_evidence, "spatial_evidence")
        if any(not isinstance(item, Mapping) for item in self.task_evidence_references):
            raise ValueError("task_evidence_references must contain mappings")
        _mapping(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskRequest:
    """Task-independent request presented to a capability."""

    task_type: str
    scene_id: str | None = None
    scene_reference: Mapping[str, Any] | None = None
    query: str | None = None
    requested_modalities: tuple[str, ...] = ()
    temporal_context: Mapping[str, Any] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        task_type = _nonempty(self.task_type, "task_type").lower()
        object.__setattr__(self, "task_type", task_type)
        if self.scene_id is None and self.scene_reference is None:
            raise ValueError("scene_id or scene_reference is required")
        if self.scene_id is not None:
            object.__setattr__(self, "scene_id", _nonempty(self.scene_id, "scene_id"))
        if self.scene_reference is not None:
            _mapping(self.scene_reference, "scene_reference")
        if self.query is not None and (not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 4000):
            raise ValueError("query must contain 1 to 4000 characters when supplied")
        if task_type in _QUERY_REQUIRED_TASKS and self.query is None:
            raise ValueError(f"{task_type} requires a query")
        modalities = _strings(self.requested_modalities, "requested_modalities")
        object.__setattr__(self, "requested_modalities", modalities)
        required = _KNOWN_MODALITY_REQUIREMENTS.get(task_type, frozenset())
        if not required.issubset(modalities):
            raise ValueError(f"{task_type} requires modalities: {', '.join(sorted(required))}")
        if task_type in _TEMPORAL_TASKS and self.temporal_context is None:
            raise ValueError(f"{task_type} requires temporal_context")
        if self.temporal_context is not None:
            _mapping(self.temporal_context, "temporal_context")
        for name in ("parameters", "constraints", "execution_metadata"):
            _mapping(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskResult:
    """Generic result whose confidence and evidence are explicitly optional."""

    status: str
    task_type: str
    output: Any = None
    evidence: Mapping[str, Any] | None = None
    confidence: float | None = None
    confidence_calibrated: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[Mapping[str, Any], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"status must be one of {sorted(RESULT_STATUSES)}")
        object.__setattr__(self, "task_type", _nonempty(self.task_type, "task_type").lower())
        if type(self.confidence_calibrated) is not bool:
            raise ValueError("confidence_calibrated must be boolean")
        if self.confidence is not None:
            if type(self.confidence) not in {int, float} or not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
                raise ValueError("confidence must be a finite number from 0 to 1")
            if not self.confidence_calibrated:
                raise ValueError("non-null confidence must be explicitly calibrated")
        if self.evidence is not None:
            _mapping(self.evidence, "evidence")
        _mapping(self.provenance, "provenance")
        if any(not isinstance(item, Mapping) for item in self.artifacts):
            raise ValueError("artifacts must contain mappings")
        _mapping(self.diagnostics, "diagnostics")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Capability:
    """Discovery metadata only; it does not contain executable plugin code."""

    name: str
    task_type: str
    version: str
    supported_modalities: frozenset[str] = frozenset()
    supported_input_types: frozenset[str] = frozenset({"satellite_scene"})
    supports_temporal: bool = False
    supports_spatial_evidence: bool = False
    supports_confidence: bool = False
    input_contract: str = "TaskRequest"
    output_contract: str = "TaskResult"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        object.__setattr__(self, "task_type", _nonempty(self.task_type, "task_type").lower())
        object.__setattr__(self, "version", _nonempty(self.version, "version"))
        object.__setattr__(self, "supported_modalities", frozenset(_strings(self.supported_modalities, "supported_modalities")))
        object.__setattr__(self, "supported_input_types", frozenset(_strings(self.supported_input_types, "supported_input_types")))
        if not self.supported_input_types:
            raise ValueError("supported_input_types must not be empty")
        for name in ("supports_temporal", "supports_spatial_evidence", "supports_confidence"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        _nonempty(self.input_contract, "input_contract")
        _nonempty(self.output_contract, "output_contract")

    def supports(self, request: TaskRequest, scene: SceneBundle | None = None) -> bool:
        if request.task_type != self.task_type:
            return False
        if not set(request.requested_modalities).issubset(self.supported_modalities):
            return False
        if request.temporal_context is not None and not self.supports_temporal:
            return False
        if scene is not None:
            if scene.input_type.lower() not in self.supported_input_types:
                return False
            if not set(request.requested_modalities).issubset(scene.available_modalities):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible discovery metadata."""
        return {
            "name": self.name,
            "task_type": self.task_type,
            "version": self.version,
            "supported_modalities": sorted(self.supported_modalities),
            "supported_input_types": sorted(self.supported_input_types),
            "supports_temporal": self.supports_temporal,
            "supports_spatial_evidence": self.supports_spatial_evidence,
            "supports_confidence": self.supports_confidence,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
        }


class CapabilityRegistry:
    """Small in-memory registry for discovery and structural matching."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if not isinstance(capability, Capability):
            raise TypeError("capability must be a Capability")
        if capability.name in self._capabilities:
            raise ValueError(f"capability already registered: {capability.name}")
        self._capabilities[capability.name] = capability

    def lookup(self, name: str) -> Capability | None:
        return self._capabilities.get(name)

    def require(self, name: str) -> Capability:
        """Return a named capability or reject an unknown name explicitly."""
        capability = self.lookup(name)
        if capability is None:
            raise LookupError(f"unknown capability: {name}")
        return capability

    def lookup_task_type(self, task_type: str) -> Capability | None:
        """Return the sole capability for a task type, rejecting ambiguity."""
        normalized = _nonempty(task_type, "task_type").lower()
        matches = tuple(
            capability for capability in self.list_capabilities()
            if capability.task_type == normalized
        )
        if len(matches) > 1:
            raise LookupError(f"multiple capabilities registered for task type: {normalized}")
        return matches[0] if matches else None

    def require_task_type(self, task_type: str) -> Capability:
        """Return a capability by task type or reject an unsupported task."""
        capability = self.lookup_task_type(task_type)
        if capability is None:
            raise LookupError(f"unknown task type: {task_type}")
        return capability

    def list_capabilities(self) -> tuple[Capability, ...]:
        return tuple(self._capabilities[name] for name in sorted(self._capabilities))

    def matching(self, request: TaskRequest, scene: SceneBundle | None = None) -> tuple[Capability, ...]:
        return tuple(capability for capability in self.list_capabilities() if capability.supports(request, scene))

    def supports(self, request: TaskRequest, scene: SceneBundle | None = None) -> bool:
        return bool(self.matching(request, scene))


@dataclass(frozen=True)
class AdaptedAnalysis:
    """Legacy result plus its non-mutating canonical views."""

    legacy_result: Mapping[str, Any]
    scene: SceneBundle
    task_result: TaskResult


def _live_representation_refs(
    *, analysis_id: str, scene_id: str, modalities: tuple[str, ...],
    cube: Mapping[str, Any], features: Mapping[str, Any],
    spatial: Mapping[str, Any], crs: str | None,
) -> tuple[RepresentationRef, ...]:
    """Describe live values truthfully without persisting or recomputing them."""
    run_reference = f"run:{analysis_id}"
    references: list[RepresentationRef] = []
    dimensions = tuple(cube.get("spatial_dimensions") or ())
    for modality, channels, kind in (
        ("optical", 12, RepresentationType.RAW_OPTICAL),
        ("sar", 2, RepresentationType.RAW_SAR),
    ):
        if modality in modalities and len(dimensions) == 2:
            references.append(RepresentationRef(
                reference_id=f"{run_reference}:{kind.value}", scene_id=scene_id,
                representation_type=kind, modality=modality,
                shape=(channels, *dimensions), dtype="float32",
                spatial_reference=SpatialReference(level="pixel", crs=crs),
                producer="raster_inputs.assemble", provenance_reference=run_reference,
                status="not_addressable",
            ))
    physical = features.get("spectral") or {}
    if physical.get("status") == "computed":
        references.append(RepresentationRef(
            reference_id=f"{run_reference}:physical_features", scene_id=scene_id,
            representation_type=RepresentationType.PHYSICAL_FEATURES,
            modality="optical_sar" if len(modalities) == 2 else (modalities[0] if modalities else None),
            version=physical.get("schema"), shape=(int(physical["dimension"]),), dtype="float32",
            spatial_reference=SpatialReference(level="feature"),
            producer="modality_features.extract", provenance_reference=run_reference,
            status="materialized",
        ))
    deep = features.get("deep") or {}
    croma_types = {
        "optical": (RepresentationType.OPTICAL_CROMA, "optical"),
        "SAR": (RepresentationType.SAR_CROMA, "sar"),
        "joint": (RepresentationType.JOINT_CROMA, "optical_sar"),
    }
    for key, shape in (deep.get("representations") or {}).items():
        stem = key.removesuffix("_encodings").removesuffix("_GAP")
        if stem not in croma_types:
            continue
        kind, modality = croma_types[stem]
        is_token = key.endswith("_encodings")
        references.append(RepresentationRef(
            reference_id=f"{run_reference}:{key}", scene_id=scene_id,
            representation_type=kind, modality=modality, shape=tuple(shape), dtype="float32",
            spatial_reference=SpatialReference(
                level="token" if is_token else "scene", crs=crs if is_token else None,
                token_grid=(15, 15) if is_token else None,
                mapping_reference="spatial_evidence.scene.token_grid" if is_token and spatial.get("status") == "AVAILABLE" else None,
                mapping_version="north_up_row_major_120_to_15_v1" if is_token and spatial.get("status") == "AVAILABLE" else None,
            ),
            producer="CROMAAdapter", producer_version=deep.get("source"),
            preprocessing_version="official CROMA mean +/- 2 std clipping",
            provenance_reference=run_reference, status="not_addressable",
        ))
    if deep.get("status") == "computed" and deep.get("pooled_dimension"):
        references.append(RepresentationRef(
            reference_id=f"{run_reference}:croma_scene", scene_id=scene_id,
            representation_type=RepresentationType.CROMA_SCENE,
            modality="optical_sar" if len(modalities) == 2 else (modalities[0] if modalities else None),
            shape=(int(deep["pooled_dimension"]),), dtype="float32",
            spatial_reference=SpatialReference(level="scene"),
            producer="analysis_engine._deep_features", producer_version=deep.get("source"),
            preprocessing_version="official CROMA mean +/- 2 std clipping",
            provenance_reference=run_reference, status="materialized",
        ))
    hybrid = features.get("hybrid") or {}
    if hybrid.get("status") == "computed":
        references.append(RepresentationRef(
            reference_id=f"{run_reference}:hybrid", scene_id=scene_id,
            representation_type=RepresentationType.HYBRID, modality="optical_sar",
            shape=(int(hybrid["dimension"]),), dtype="float32",
            spatial_reference=SpatialReference(level="scene"), producer="HybridFusion",
            producer_version=hybrid.get("source"), provenance_reference=run_reference,
            status="materialized",
        ))
    if spatial.get("status") == "AVAILABLE":
        common = {
            "scene_id": scene_id,
            "modality": "optical_sar" if len(modalities) == 2 else (modalities[0] if modalities else None),
            "producer": "evidence_schema.build_spatial_evidence",
            "version": spatial.get("schema_version"),
            "provenance_reference": run_reference,
            "status": "materialized",
        }
        references.extend((
            RepresentationRef(
                reference_id=f"{run_reference}:regions", representation_type=RepresentationType.REGIONS,
                shape=((len(spatial.get("regions") or ()),) if spatial.get("regions") else None), dtype="json",
                spatial_reference=SpatialReference(level="region", crs=crs, mapping_reference="spatial_evidence.scene"),
                **common,
            ),
            RepresentationRef(
                reference_id=f"{run_reference}:evidence", representation_type=RepresentationType.EVIDENCE,
                dtype="json", spatial_reference=SpatialReference(level="evidence", crs=crs, mapping_reference="spatial_evidence.scene"),
                **common,
            ),
        ))
    references.append(RepresentationRef(
        reference_id=f"{run_reference}:metadata", scene_id=scene_id,
        representation_type=RepresentationType.METADATA, dtype="json",
        spatial_reference=SpatialReference(level="metadata", crs=crs),
        producer="MultimodalCube.summary", provenance_reference=run_reference,
        status="materialized" if cube else "unavailable",
    ))
    return tuple(references)


def adapt_run_analysis_output(result: Mapping[str, Any], request: Mapping[str, Any] | None = None) -> AdaptedAnalysis:
    """Wrap an existing ``run_analysis`` result without recomputing any value."""
    if not isinstance(result, Mapping):
        raise ValueError("run_analysis output must be a mapping")
    analysis_id = _nonempty(result.get("analysis_id"), "analysis_id")
    plan = result.get("query_interpretation") or {}
    retrieval = result.get("data_retrieval") or {}
    cube = result.get("data_cube") or {}
    metadata = cube.get("metadata") or {}
    spatial = result.get("spatial_evidence") or {}
    spatial_available = spatial.get("status") == "AVAILABLE"
    spatial_scene = spatial.get("scene") or {}
    requested = request or {}
    modalities = tuple(cube.get("modalities") or plan.get("modalities") or ())
    reference_metadata = metadata.get("optical") or metadata.get("sar") or {}
    identifiers = {str(key): str(value) for key, value in (retrieval.get("assets") or {}).items()}
    if requested.get("sample_id") is not None:
        identifiers.setdefault("sample_id", str(requested["sample_id"]))
    scene_id = str(spatial_scene.get("sample_id") or identifiers.get("sample_id") or analysis_id)
    compatibility = ((result.get("validation") or {}).get("checks") or {}).get("optical_sar_compatibility")
    if len(modalities) < 2:
        coregistration = "not_applicable"
    elif isinstance(compatibility, Mapping):
        coregistration = "aligned" if compatibility and all(compatibility.values()) else "misaligned"
    else:
        coregistration = "unknown"
    features = result.get("features") or {}
    crs = spatial_scene.get("crs") or reference_metadata.get("crs")
    representations = RepresentationSet(
        physical_features=features.get("spectral"),
        croma_scene=features.get("deep"),
        joint=features.get("hybrid"),
        regions=spatial.get("regions") if spatial_available else None,
        references=_live_representation_refs(
            analysis_id=analysis_id,
            scene_id=scene_id,
            modalities=modalities,
            cube=cube,
            features=features,
            spatial=spatial,
            crs=crs,
        ),
    )
    provenance = {
        "run_id": analysis_id,
        "input_identifiers": identifiers,
        "source": retrieval.get("source"),
        "preprocessing": result.get("preprocessing"),
        "representation": {
            "physical_schema": (features.get("spectral") or {}).get("schema"),
            "croma_source": (features.get("deep") or {}).get("source"),
        },
        "task": {"type": plan.get("task"), "version": result.get("schema_version")},
        "execution": {"timestamp": result.get("timestamp"), "trace": result.get("execution_trace")},
        "evidence": spatial.get("provenance") if spatial_available else None,
    }
    evidence = spatial if spatial_available else None
    scene = SceneBundle(
        scene_id=scene_id,
        input_identifiers=identifiers,
        analysis_id=analysis_id,
        available_modalities=modalities,
        temporal_available=bool(result.get("temporal", {}).get("status") not in {None, "not requested", "unavailable", "requires valid input"}),
        temporal_context=(requested.get("temporal_context") if isinstance(requested.get("temporal_context"), Mapping) else None),
        crs=crs,
        bounds=tuple(reference_metadata["bounds"]) if reference_metadata.get("bounds") else None,
        resolution=tuple(reference_metadata["resolution"]) if reference_metadata.get("resolution") else None,
        dimensions=tuple(spatial_scene.get("image_shape") or cube.get("spatial_dimensions")) if (spatial_scene.get("image_shape") or cube.get("spatial_dimensions")) else None,
        georeferencing_status="validated" if reference_metadata.get("crs") and reference_metadata.get("transform") else "unavailable",
        coregistration_status=coregistration,
        representations=representations,
        spatial_evidence=evidence,
        task_evidence_references=tuple(result.get("evidence") or ()),
        provenance=provenance,
    )
    status = {"completed": "success", "partial": "partial", "rejected": "failure", "unavailable": "unavailable"}.get(result.get("status"), "failure")
    task_type = str(plan.get("task") or requested.get("analysis_type") or "classification")
    task_result = TaskResult(
        status=status,
        task_type=task_type,
        output={
            "analysis": result,
            "model_results": result.get("model_results"),
            "features": features,
            "interpretation": result.get("interpretation"),
            "explanation": result.get("llm_explanation"),
        },
        evidence=evidence,
        confidence=None,
        confidence_calibrated=False,
        provenance=provenance,
        artifacts=tuple(result.get("evidence") or ()),
        diagnostics={
            "legacy_status": result.get("status"),
            "legacy_confidence": result.get("confidence"),
            "validation": result.get("validation"),
            "warnings": result.get("warnings"),
            "error": result.get("error"),
        },
    )
    return AdaptedAnalysis(legacy_result=result, scene=scene, task_result=task_result)


def run_analysis_with_contracts(request: Mapping[str, Any]) -> AdaptedAnalysis:
    """Call the one canonical analysis entry, then create contract views."""
    from .analysis_engine import run_analysis

    legacy_result = run_analysis(dict(request))
    return adapt_run_analysis_output(legacy_result, request)
