"""Evidence-only RSVQA source gate; does not download or parse benchmark data."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .contracts import VerificationStatus
from .provenance import result_artifact_hash


class SourceAuthority(str, Enum):
    PRIMARY_AUTHORITATIVE = "PRIMARY_AUTHORITATIVE"
    AUTHOR_LINKED_MIRROR = "AUTHOR_LINKED_MIRROR"
    SECONDARY_MIRROR = "SECONDARY_MIRROR"
    DERIVED_SUBSET = "DERIVED_SUBSET"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class RSVQASourceEvidence:
    variant: str
    authority: SourceAuthority
    revision: str | None
    revision_type: str | None
    archive_identifier: str | None
    file_checksums: Mapping[str, str]
    annotations_license: VerificationStatus
    imagery_license: VerificationStatus
    underlying_imagery_license: VerificationStatus
    code_evaluator_revision: str | None = None

    def revision_verified(self) -> bool:
        return self.authority is SourceAuthority.PRIMARY_AUTHORITATIVE and bool(self.revision and self.revision_type and self.archive_identifier and self.file_checksums)

    def acquisition_permitted(self) -> bool:
        return self.revision_verified() and self.annotations_license is VerificationStatus.VERIFIED and self.imagery_license is VerificationStatus.VERIFIED and self.underlying_imagery_license is VerificationStatus.VERIFIED

    def blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.revision_verified(): reasons.append("SOURCE_REVISION_UNVERIFIED")
        if self.annotations_license is not VerificationStatus.VERIFIED: reasons.append("ANNOTATION_LICENSE_UNVERIFIED")
        if self.imagery_license is not VerificationStatus.VERIFIED: reasons.append("IMAGE_LICENSE_UNVERIFIED")
        if self.underlying_imagery_license is not VerificationStatus.VERIFIED: reasons.append("UNDERLYING_IMAGERY_LICENSE_UNVERIFIED")
        return tuple(reasons)


def deterministic_acquisition_manifest(source: RSVQASourceEvidence, downloaded: tuple[Mapping[str, object], ...] = ()) -> dict[str, object]:
    """Schema for a bounded receipt; an empty list truthfully records no acquisition."""
    rows = tuple(dict(row) for row in downloaded)
    body = {"run_kind": "BOUNDED_SOURCE_AUDIT", "variant": source.variant, "authority": source.authority.value, "source_revision": source.revision, "source_revision_type": source.revision_type, "archive_identifier": source.archive_identifier, "license_gate": "PASS" if source.acquisition_permitted() else "BLOCKED", "downloaded_artifacts": rows}
    body["manifest_sha256"] = result_artifact_hash(body)
    return body
