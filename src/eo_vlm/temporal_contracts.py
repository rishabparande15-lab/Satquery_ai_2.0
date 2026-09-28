"""Fail-closed temporal input and representation contracts.

These contracts preserve two independently identified observations at a
temporal boundary.  They do not calculate pixel differences, infer change, or
materialize learned temporal features.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping

from ..architecture_contracts import RepresentationRef


TEMPORAL_CONTRACT_VERSION = "phase3j3_temporal_contract_v1"
NOT_COMPUTED = "NOT_COMPUTED"
UNKNOWN = "UNKNOWN"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized = dict(value)
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be deterministic JSON-compatible metadata") from exc
    return normalized


class TemporalOrderStatus(str, Enum):
    VERIFIED = "TEMPORAL_ORDER_VERIFIED"
    UNKNOWN = "TEMPORAL_ORDER_UNKNOWN"


class SpatialCorrespondenceStatus(str, Enum):
    VERIFIED = "SPATIAL_CORRESPONDENCE_VERIFIED"
    UNKNOWN = "SPATIAL_CORRESPONDENCE_UNKNOWN"


class SpatialCorrespondence(str, Enum):
    COREGISTERED = "COREGISTERED"
    SPATIALLY_CORRESPONDING = "SPATIALLY_CORRESPONDING"
    SCENE_CORRESPONDING = "SCENE_CORRESPONDING"
    UNKNOWN = "UNKNOWN"


class TemporalModality(str, Enum):
    S2 = "s2"
    S1 = "s1"
    OPTICAL = "optical"
    SAR = "sar"


class TemporalEvidenceKind(str, Enum):
    OBSERVED_T1 = "OBSERVED_T1"
    OBSERVED_T2 = "OBSERVED_T2"
    TEMPORAL_MODEL_EVIDENCE = "TEMPORAL_MODEL_EVIDENCE"
    MODEL_LANGUAGE_OUTPUT = "MODEL_LANGUAGE_OUTPUT"
    SOURCE_METADATA = "SOURCE_METADATA"


@dataclass(frozen=True)
class TemporalProvenance:
    """Pinned source identity; source metadata is never promoted to evidence."""

    source_id: str
    source_revision: str | None = None
    provenance_reference: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        for name in ("source_revision", "provenance_reference"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        object.__setattr__(self, "attributes", _json_mapping(self.attributes, "attributes"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalMetadata:
    """Optional time metadata, validated only when supplied."""

    t1_timestamp: str | None = None
    t2_timestamp: str | None = None
    interval_seconds: float | None = None

    def __post_init__(self) -> None:
        parsed: list[datetime | None] = []
        for name in ("t1_timestamp", "t2_timestamp"):
            value = getattr(self, name)
            if value is None:
                parsed.append(None)
                continue
            normalized = _text(value, name)
            try:
                parsed.append(datetime.fromisoformat(normalized.replace("Z", "+00:00")))
            except ValueError as exc:
                raise ValueError(f"{name} must be ISO-8601 when supplied") from exc
            object.__setattr__(self, name, normalized)
        if (parsed[0] is None) != (parsed[1] is None):
            raise ValueError("timestamps must be supplied together")
        if parsed[0] is not None and parsed[0] > parsed[1]:
            raise ValueError("t1_timestamp must not be after t2_timestamp")
        if self.interval_seconds is not None:
            if not isinstance(self.interval_seconds, (int, float)) or isinstance(self.interval_seconds, bool) or self.interval_seconds < 0:
                raise ValueError("interval_seconds must be a non-negative number")
            if parsed[0] is None:
                raise ValueError("interval_seconds requires both timestamps")


@dataclass(frozen=True)
class TemporalFrame:
    """One observed timepoint; payloads stay outside this metadata contract."""

    reference_id: str
    modality: TemporalModality | str
    provenance: TemporalProvenance
    tensor_shape: tuple[int, ...] | None = None
    representation_ref: RepresentationRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _text(self.reference_id, "reference_id"))
        try:
            object.__setattr__(self, "modality", TemporalModality(self.modality))
        except ValueError as exc:
            raise ValueError("unsupported temporal modality") from exc
        if not isinstance(self.provenance, TemporalProvenance):
            raise ValueError("provenance must be TemporalProvenance")
        if self.tensor_shape is not None:
            if not self.tensor_shape or any(type(axis) is not int or axis <= 0 for axis in self.tensor_shape):
                raise ValueError("tensor_shape must contain positive integers")
            expected = {TemporalModality.S2: (12, 120, 120), TemporalModality.S1: (2, 120, 120)}.get(self.modality)
            if expected is not None and self.tensor_shape != expected:
                raise ValueError(f"{self.modality.value} tensor_shape must be {expected}")
        if self.representation_ref is not None and not isinstance(self.representation_ref, RepresentationRef):
            raise ValueError("representation_ref must be RepresentationRef")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "reference_id": self.reference_id,
            "modality": self.modality.value,
            "provenance": self.provenance.to_dict(),
            "tensor_shape": list(self.tensor_shape) if self.tensor_shape else None,
            "metadata": dict(self.metadata),
        }
        if self.representation_ref is not None:
            result["representation_ref"] = self.representation_ref.to_dict()
        return result


@dataclass(frozen=True)
class TemporalInput:
    """An ordered pair with explicit, conservatively stated correspondence."""

    t1: TemporalFrame
    t2: TemporalFrame
    temporal_order: TemporalOrderStatus | str
    spatial_status: SpatialCorrespondenceStatus | str
    spatial_correspondence: SpatialCorrespondence | str = SpatialCorrespondence.UNKNOWN
    temporal_metadata: TemporalMetadata = field(default_factory=TemporalMetadata)
    provenance: TemporalProvenance | None = None
    validation_status: str = "VALID"

    def __post_init__(self) -> None:
        if not isinstance(self.t1, TemporalFrame) or not isinstance(self.t2, TemporalFrame):
            raise ValueError("t1 and t2 must be TemporalFrame values")
        if self.t1.reference_id == self.t2.reference_id:
            raise ValueError("t1 and t2 must preserve distinct timepoint identities")
        try:
            object.__setattr__(self, "temporal_order", TemporalOrderStatus(self.temporal_order))
            object.__setattr__(self, "spatial_status", SpatialCorrespondenceStatus(self.spatial_status))
            object.__setattr__(self, "spatial_correspondence", SpatialCorrespondence(self.spatial_correspondence))
        except ValueError as exc:
            raise ValueError("invalid temporal or spatial status") from exc
        if not isinstance(self.temporal_metadata, TemporalMetadata):
            raise ValueError("temporal_metadata must be TemporalMetadata")
        if self.spatial_status is SpatialCorrespondenceStatus.VERIFIED and self.spatial_correspondence is SpatialCorrespondence.UNKNOWN:
            raise ValueError("verified spatial correspondence cannot be UNKNOWN")
        if self.spatial_status is SpatialCorrespondenceStatus.UNKNOWN and self.spatial_correspondence is not SpatialCorrespondence.UNKNOWN:
            raise ValueError("unverified spatial correspondence must be UNKNOWN")
        if self.provenance is not None and not isinstance(self.provenance, TemporalProvenance):
            raise ValueError("provenance must be TemporalProvenance when supplied")
        object.__setattr__(self, "validation_status", _text(self.validation_status, "validation_status"))

    @property
    def modalities(self) -> tuple[str, str]:
        return self.t1.modality.value, self.t2.modality.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": TEMPORAL_CONTRACT_VERSION,
            "t1": self.t1.to_dict(), "t2": self.t2.to_dict(),
            "temporal_order": self.temporal_order.value,
            "spatial_status": self.spatial_status.value,
            "spatial_correspondence": self.spatial_correspondence.value,
            "temporal_metadata": asdict(self.temporal_metadata),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "validation_status": self.validation_status,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemporalSequenceInput:
    """Future-facing sequence contract; it does not encode a sequence."""

    frames: tuple[TemporalFrame, ...]
    provenance: TemporalProvenance

    def __post_init__(self) -> None:
        if len(self.frames) < 2 or any(not isinstance(frame, TemporalFrame) for frame in self.frames):
            raise ValueError("frames must contain at least two TemporalFrame values")
        identities = tuple(frame.reference_id for frame in self.frames)
        if len(identities) != len(set(identities)):
            raise ValueError("sequence frame identities must be unique")
        if not isinstance(self.provenance, TemporalProvenance):
            raise ValueError("provenance must be TemporalProvenance")


@dataclass(frozen=True)
class TemporalResourceEstimate:
    estimated_image_count: int
    estimated_vram_bytes: str = UNKNOWN
    estimated_sequence_cost: str = UNKNOWN
    estimated_generation_cost: str = UNKNOWN

    def __post_init__(self) -> None:
        if type(self.estimated_image_count) is not int or self.estimated_image_count < 2:
            raise ValueError("estimated_image_count must be at least two")
        for name in ("estimated_vram_bytes", "estimated_sequence_cost", "estimated_generation_cost"):
            if getattr(self, name) != UNKNOWN:
                raise ValueError(f"{name} must be UNKNOWN until a measured implementation exists")


@dataclass(frozen=True)
class TemporalRepresentation:
    """Representation boundary with no fabricated temporal or change features."""

    temporal_input: TemporalInput
    t1_representation_ref: RepresentationRef | None = None
    t2_representation_ref: RepresentationRef | None = None
    temporal_features: str = NOT_COMPUTED
    change_features: str = NOT_COMPUTED
    resource_estimate: TemporalResourceEstimate = field(default_factory=lambda: TemporalResourceEstimate(2))

    def __post_init__(self) -> None:
        if not isinstance(self.temporal_input, TemporalInput):
            raise ValueError("temporal_input must be TemporalInput")
        for name in ("t1_representation_ref", "t2_representation_ref"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, RepresentationRef):
                raise ValueError(f"{name} must be RepresentationRef when supplied")
        if self.temporal_features != NOT_COMPUTED or self.change_features != NOT_COMPUTED:
            raise ValueError("temporal and change features must remain NOT_COMPUTED before a validated model exists")
        if not isinstance(self.resource_estimate, TemporalResourceEstimate):
            raise ValueError("resource_estimate must be TemporalResourceEstimate")


class TemporalFusionUnavailable(RuntimeError):
    """Raised instead of silently creating a temporal/change representation."""


class TemporalFusionAdapter(ABC):
    """Model-swappable future fusion boundary; no model is selected here."""

    @abstractmethod
    def get_capabilities(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_provenance(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def estimate_resources(self, temporal_input: TemporalInput) -> TemporalResourceEstimate:
        raise NotImplementedError

    def encode_pair(self, temporal_input: TemporalInput) -> TemporalRepresentation:
        raise TemporalFusionUnavailable("learned temporal fusion is not implemented")

    def encode_sequence(self, sequence: TemporalSequenceInput) -> TemporalRepresentation:
        raise TemporalFusionUnavailable("learned temporal sequence fusion is not implemented")


class ContractOnlyTemporalFusionAdapter(TemporalFusionAdapter):
    """Inspectable adapter that exposes the boundary and refuses inference."""

    def get_capabilities(self) -> Mapping[str, Any]:
        return {
            "model_bound": False, "learned_temporal_encoder": False,
            "learned_temporal_fusion": False, "change_features": NOT_COMPUTED,
            "supported_modalities": [item.value for item in TemporalModality],
        }

    def get_provenance(self) -> Mapping[str, Any]:
        return {"contract_version": TEMPORAL_CONTRACT_VERSION, "implementation": "contract_only", "model": None}

    def estimate_resources(self, temporal_input: TemporalInput) -> TemporalResourceEstimate:
        if not isinstance(temporal_input, TemporalInput):
            raise ValueError("temporal_input must be TemporalInput")
        return TemporalResourceEstimate(estimated_image_count=2)


@dataclass(frozen=True)
class TemporalEvidenceItem:
    kind: TemporalEvidenceKind | str
    reference_id: str
    provenance: TemporalProvenance

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", TemporalEvidenceKind(self.kind))
        except ValueError as exc:
            raise ValueError("invalid temporal evidence kind") from exc
        object.__setattr__(self, "reference_id", _text(self.reference_id, "reference_id"))
        if not isinstance(self.provenance, TemporalProvenance):
            raise ValueError("provenance must be TemporalProvenance")
