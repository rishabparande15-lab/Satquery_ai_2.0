"""Truthful answer summaries without model-mediated reconciliation."""
from dataclasses import dataclass
from .evidence import EvidenceType
from .reconciler import Reconciliation
from .query_types import ExecutionStatus


@dataclass(frozen=True)
class AnswerAssembly:
    status: ExecutionStatus
    answer_kind: str
    text: str


class AnswerAssembler:
    def assemble(self, reconciliation: Reconciliation, blocked_reason: str | None = None) -> AnswerAssembly:
        if blocked_reason:
            return AnswerAssembly(ExecutionStatus.BLOCKED, "neither", blocked_reason)
        if reconciliation.conflicts:
            return AnswerAssembly(ExecutionStatus.CONFLICT, "conflict", "CONFLICT_DETECTED: claims are preserved without an automatic resolution.")
        kinds = {item.evidence_type for item in reconciliation.evidence}
        scientific = EvidenceType.SCIENTIFIC_PREDICTION in kinds
        language = EvidenceType.MODEL_LANGUAGE_OUTPUT in kinds
        kind = "both" if scientific and language else "scientific_prediction" if scientific else "model_language" if language else "neither"
        return AnswerAssembly(ExecutionStatus.EXECUTED if kinds else ExecutionStatus.NOT_VERIFIED, kind, "Evidence collected; see provenance and typed evidence." if kinds else "No executable evidence was produced.")
