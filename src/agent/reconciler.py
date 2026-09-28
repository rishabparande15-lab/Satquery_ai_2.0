"""Deterministic conflict detection; it intentionally does not resolve conflicts."""
from dataclasses import dataclass
from .evidence import Evidence
from .query_types import ExecutionStatus


@dataclass(frozen=True)
class EvidenceConflict:
    claim_key: str
    evidence_ids: tuple[str, ...]
    claims: tuple[object, ...]


@dataclass(frozen=True)
class Reconciliation:
    evidence: tuple[Evidence, ...]
    conflicts: tuple[EvidenceConflict, ...]
    status: ExecutionStatus


class EvidenceReconciler:
    def reconcile(self, evidence: tuple[Evidence, ...]) -> Reconciliation:
        grouped: dict[str, list[Evidence]] = {}
        for item in evidence:
            grouped.setdefault(item.claim_key, []).append(item)
        conflicts = []
        for key, items in grouped.items():
            claims = tuple(item.claim for item in items)
            if len({repr(value) for value in claims}) > 1:
                conflicts.append(EvidenceConflict(key, tuple(item.evidence_id for item in items), claims))
        return Reconciliation(evidence, tuple(conflicts), ExecutionStatus.CONFLICT if conflicts else ExecutionStatus.EXECUTED)
