"""RSVQA source-acceptance contract; it records evidence and never acquires data."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

from .contracts import DatasetReadiness, VerificationStatus


class AcceptanceState(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class RSVQADatasetAcceptance:
    dataset_identity: AcceptanceState = AcceptanceState.UNVERIFIED
    dataset_version_or_revision: AcceptanceState = AcceptanceState.UNVERIFIED
    image_manifest: AcceptanceState = AcceptanceState.MISSING
    question_manifest: AcceptanceState = AcceptanceState.MISSING
    answer_manifest: AcceptanceState = AcceptanceState.MISSING
    image_question_linkage: AcceptanceState = AcceptanceState.UNVERIFIED
    question_answer_linkage: AcceptanceState = AcceptanceState.UNVERIFIED
    official_train_split: AcceptanceState = AcceptanceState.MISSING
    official_validation_split: AcceptanceState = AcceptanceState.MISSING
    official_test_split: AcceptanceState = AcceptanceState.MISSING
    licensing_evidence: AcceptanceState = AcceptanceState.UNVERIFIED
    provenance: AcceptanceState = AcceptanceState.UNVERIFIED
    checksums_or_equivalent_artifact_identity: AcceptanceState = AcceptanceState.UNVERIFIED
    duplicate_check: AcceptanceState = AcceptanceState.UNVERIFIED
    cross_split_leakage_check: AcceptanceState = AcceptanceState.UNVERIFIED
    near_duplicate_check_if_applicable: AcceptanceState = AcceptanceState.UNVERIFIED
    modality_definition: AcceptanceState = AcceptanceState.UNVERIFIED

    def blocking_reasons(self) -> tuple[str, ...]:
        allowed_na = {"near_duplicate_check_if_applicable"}
        return tuple(
            f"RSVQA_{item.name.upper()}_{getattr(self, item.name).value}"
            for item in fields(self)
            if getattr(self, item.name) is not AcceptanceState.VERIFIED
            and not (item.name in allowed_na and getattr(self, item.name) is AcceptanceState.NOT_APPLICABLE)
        )

    def to_dataset_readiness(self) -> DatasetReadiness:
        verified = AcceptanceState.VERIFIED
        return DatasetReadiness(
            source="RSVQA" if self.dataset_identity is verified else "UNKNOWN",
            dataset_revision="RSVQA_VERIFIED_REVISION" if self.dataset_version_or_revision is verified else "UNKNOWN",
            benchmark_revision="RSVQA_VERIFIED_BENCHMARK_REVISION" if self.dataset_version_or_revision is verified else "UNKNOWN",
            image_availability=VerificationStatus.VERIFIED if self.image_manifest is verified else VerificationStatus.UNAVAILABLE,
            annotation_availability=VerificationStatus.VERIFIED if self.question_manifest is verified and self.answer_manifest is verified else VerificationStatus.UNAVAILABLE,
            sample_linkage=VerificationStatus.VERIFIED if self.image_question_linkage is verified and self.question_answer_linkage is verified else VerificationStatus.UNVERIFIED,
            license_status=VerificationStatus.VERIFIED if self.licensing_evidence is verified else VerificationStatus.UNVERIFIED,
            provenance_status=VerificationStatus.VERIFIED if self.provenance is verified and self.checksums_or_equivalent_artifact_identity is verified else VerificationStatus.UNVERIFIED,
            leakage_status=VerificationStatus.VERIFIED if self.duplicate_check is verified and self.cross_split_leakage_check is verified and self.near_duplicate_check_if_applicable in {verified, AcceptanceState.NOT_APPLICABLE} else VerificationStatus.UNVERIFIED,
            modality_compatibility=VerificationStatus.VERIFIED if self.modality_definition is verified else VerificationStatus.UNVERIFIED,
            split_manifest={"train": (), "validation": (), "test": ()} if all(value is verified for value in (self.official_train_split, self.official_validation_split, self.official_test_split)) else {},
        )


def rsvqa_acceptance_schema() -> tuple[str, ...]:
    return tuple(item.name for item in fields(RSVQADatasetAcceptance))
