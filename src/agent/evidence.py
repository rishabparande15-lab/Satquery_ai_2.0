"""Evidence is typed so language never silently becomes scientific evidence."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EvidenceType(str, Enum):
    SCIENTIFIC_PREDICTION = "SCIENTIFIC_PREDICTION"
    MODEL_LANGUAGE_OUTPUT = "MODEL_LANGUAGE_OUTPUT"
    PIXEL_EVIDENCE = "PIXEL_EVIDENCE"
    REGION_EVIDENCE = "REGION_EVIDENCE"
    TEMPORAL_EVIDENCE = "TEMPORAL_EVIDENCE"
    METADATA_EVIDENCE = "METADATA_EVIDENCE"
    SOURCE_ANNOTATION = "SOURCE_ANNOTATION"


class ConfidenceSource(str, Enum):
    MODEL_PROVIDED = "MODEL_PROVIDED"
    CALIBRATED = "CALIBRATED"
    SCIENTIFIC_METRIC = "SCIENTIFIC_METRIC"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    evidence_type: EvidenceType
    source: str
    claim: Any
    claim_key: str
    modality: str | None = None
    representation: str | None = None
    spatial_reference: Mapping[str, Any] | None = None
    temporal_reference: Mapping[str, Any] | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation_status: str = "NOT_VERIFIED"

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source or not self.claim_key:
            raise ValueError("evidence_id, source, and claim_key are required")
        if self.confidence is not None and self.confidence_source == ConfidenceSource.UNKNOWN:
            raise ValueError("confidence requires an actual confidence source")
