"""Small trusted local catalog for verified representation receipts.

The catalog is deliberately an index, not a source of scientific values.  It
stores the existing immutable reference contracts and delegates all artifact
verification and loading to :mod:`src.representation_artifacts`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .architecture_contracts import (
    ArtifactRef,
    ArtifactType,
    RepresentationRef,
    RepresentationSet,
    RepresentationType,
    SceneBundle,
    SpatialReference,
)
from .representation_artifacts import ArtifactResolver


CATALOG_VERSION = 1


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    INVALID = "invalid"
    EXPIRED = "expired"


@dataclass(frozen=True)
class CatalogRecord:
    """Lifecycle metadata around one existing representation reference."""

    representation_ref: RepresentationRef
    receipt_ref: ArtifactRef
    created_at: str
    status: LifecycleStatus = LifecycleStatus.ACTIVE
    expires_at: str | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.representation_ref, RepresentationRef):
            raise ValueError("representation_ref must be a RepresentationRef")
        if not isinstance(self.receipt_ref, ArtifactRef):
            raise ValueError("receipt_ref must be an ArtifactRef")
        if self.receipt_ref.artifact_type is not ArtifactType.PROVENANCE_RECEIPT:
            raise ValueError("receipt_ref must identify a provenance receipt")
        try:
            object.__setattr__(self, "status", LifecycleStatus(self.status))
        except ValueError as exc:
            raise ValueError("invalid catalog lifecycle status") from exc
        _parse_timestamp(self.created_at, "created_at")
        if self.expires_at is not None:
            _parse_timestamp(self.expires_at, "expires_at")
        if self.status is LifecycleStatus.ACTIVE and self.invalid_reason is not None:
            raise ValueError("active catalog records cannot have an invalid_reason")
        if self.status is LifecycleStatus.INVALID and not self.invalid_reason:
            raise ValueError("invalid catalog records require an invalid_reason")
        reference = self.representation_ref
        artifact = reference.artifact
        if not reference.addressable or artifact is None:
            raise ValueError("catalog records require an available addressable representation")
        if artifact.artifact_type is not ArtifactType.NUMPY_ARRAY:
            raise ValueError("cataloged representation must use a NumPy artifact")
        if not artifact.checksum_sha256 or artifact.size_bytes is None:
            raise ValueError("cataloged artifacts require checksum and size metadata")
        if reference.shape is None or reference.dtype is None:
            raise ValueError("cataloged representations require shape and dtype metadata")
        if not artifact.run_reference:
            raise ValueError("cataloged artifacts require a run reference")
        if artifact.provenance_reference != self.receipt_ref.uri:
            raise ValueError("representation provenance must identify receipt_ref")
        if reference.provenance_reference != self.receipt_ref.uri:
            raise ValueError("representation reference must identify receipt_ref")

    @property
    def artifact_id(self) -> str:
        assert self.representation_ref.artifact is not None
        return self.representation_ref.artifact.artifact_id

    @property
    def artifact_ref(self) -> ArtifactRef:
        assert self.representation_ref.artifact is not None
        return self.representation_ref.artifact

    @property
    def scene_id(self) -> str:
        return self.representation_ref.scene_id

    @property
    def run_id(self) -> str:
        assert self.artifact_ref.run_reference is not None
        return self.artifact_ref.run_reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_ref": self.representation_ref.to_dict(),
            "receipt_ref": self.receipt_ref.to_dict(),
            "created_at": self.created_at,
            "status": self.status.value,
            "expires_at": self.expires_at,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True)
class ValidationResult:
    record: CatalogRecord
    valid: bool
    reason: str | None = None


def _parse_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _artifact_from_dict(value: Mapping[str, Any]) -> ArtifactRef:
    allowed = {field.name for field in fields(ArtifactRef)}
    if set(value).difference(allowed):
        raise ValueError("malformed artifact reference in catalog")
    return ArtifactRef(**dict(value))


def _spatial_from_dict(value: Mapping[str, Any] | None) -> SpatialReference | None:
    if value is None:
        return None
    allowed = {field.name for field in fields(SpatialReference)}
    if set(value).difference(allowed):
        raise ValueError("malformed spatial reference in catalog")
    normalized = dict(value)
    if normalized.get("token_grid") is not None:
        normalized["token_grid"] = tuple(normalized["token_grid"])
    return SpatialReference(**normalized)


def _representation_from_dict(value: Mapping[str, Any]) -> RepresentationRef:
    allowed = {field.name for field in fields(RepresentationRef)}
    if set(value).difference(allowed):
        raise ValueError("malformed representation reference in catalog")
    normalized = dict(value)
    normalized["artifact"] = _artifact_from_dict(normalized["artifact"])
    normalized["spatial_reference"] = _spatial_from_dict(normalized.get("spatial_reference"))
    if normalized.get("shape") is not None:
        normalized["shape"] = tuple(normalized["shape"])
    return RepresentationRef(**normalized)


def _record_from_dict(value: Mapping[str, Any]) -> CatalogRecord:
    expected = {"representation_ref", "receipt_ref", "created_at", "status", "expires_at", "invalid_reason"}
    if set(value) != expected:
        raise ValueError("malformed catalog record")
    if not isinstance(value["representation_ref"], Mapping) or not isinstance(value["receipt_ref"], Mapping):
        raise ValueError("malformed catalog record references")
    return CatalogRecord(
        representation_ref=_representation_from_dict(value["representation_ref"]),
        receipt_ref=_artifact_from_dict(value["receipt_ref"]),
        created_at=value["created_at"],
        status=value["status"],
        expires_at=value["expires_at"],
        invalid_reason=value["invalid_reason"],
    )


class ReceiptCatalog:
    """Deterministic JSON catalog with trusted typed registration only."""

    def __init__(self, path: Path, resolver: ArtifactResolver) -> None:
        self.path = Path(path)
        if not isinstance(resolver, ArtifactResolver):
            raise TypeError("resolver must be an ArtifactResolver")
        self.resolver = resolver
        self._records = self._read()

    def _read(self) -> tuple[CatalogRecord, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise ValueError("catalog path is not a regular file")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("catalog is not valid JSON") from exc
        if not isinstance(payload, Mapping) or set(payload) != {"catalog_version", "records"}:
            raise ValueError("malformed catalog document")
        if payload["catalog_version"] != CATALOG_VERSION or not isinstance(payload["records"], list):
            raise ValueError("unsupported or malformed catalog document")
        records = tuple(_record_from_dict(value) for value in payload["records"] if isinstance(value, Mapping))
        if len(records) != len(payload["records"]):
            raise ValueError("malformed catalog record")
        self._assert_unique(records)
        return records

    @staticmethod
    def _assert_unique(records: Iterable[CatalogRecord]) -> None:
        artifact_ids: dict[str, CatalogRecord] = {}
        uris: dict[str, CatalogRecord] = {}
        reference_ids: dict[str, CatalogRecord] = {}
        for record in records:
            for key, index, label in (
                (record.artifact_id, artifact_ids, "artifact_id"),
                (record.artifact_ref.uri, uris, "artifact URI"),
                (record.representation_ref.reference_id, reference_ids, "reference_id"),
            ):
                if key in index:
                    raise ValueError(f"duplicate {label} in catalog")
                index[key] = record

    def _write(self, records: tuple[CatalogRecord, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "catalog_version": CATALOG_VERSION,
            "records": [record.to_dict() for record in sorted(records, key=lambda item: item.artifact_id)],
        }
        content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._records = records

    def register(
        self,
        representation_ref: RepresentationRef,
        receipt_ref: ArtifactRef,
        *,
        expires_at: str | None = None,
    ) -> CatalogRecord:
        return self.register_many((representation_ref,), receipt_ref, expires_at=expires_at)[0]

    def register_many(
        self,
        representation_refs: Iterable[RepresentationRef],
        receipt_ref: ArtifactRef,
        *,
        expires_at: str | None = None,
    ) -> tuple[CatalogRecord, ...]:
        """Verify all typed references, then atomically add them to the index."""
        references = tuple(representation_refs)
        if not references:
            raise ValueError("registration requires at least one representation")
        if any(not isinstance(reference, RepresentationRef) for reference in references):
            raise TypeError("registration accepts only trusted RepresentationRef values")
        if not isinstance(receipt_ref, ArtifactRef):
            raise TypeError("receipt_ref must be an ArtifactRef")
        now = _utc_now().isoformat().replace("+00:00", "Z")
        candidates = tuple(CatalogRecord(reference, receipt_ref, now, expires_at=expires_at) for reference in references)
        for candidate in candidates:
            matches = [record for record in self._records if (
                record.artifact_id == candidate.artifact_id
                or record.artifact_ref.uri == candidate.artifact_ref.uri
                or record.representation_ref.reference_id == candidate.representation_ref.reference_id
            )]
            if matches and not (
                matches[0].representation_ref == candidate.representation_ref
                and matches[0].receipt_ref == candidate.receipt_ref
                and matches[0].expires_at == candidate.expires_at
            ):
                raise ValueError(f"conflicting catalog registration: {candidate.artifact_id}")
        for candidate in candidates:
            self.resolver.resolve(candidate.artifact_ref)
        self._verify_receipt(receipt_ref, references)
        for candidate in candidates:
            self.resolver.load_numpy(candidate.representation_ref)

        existing = list(self._records)
        results = []
        for candidate in candidates:
            matches = [record for record in existing if (
                record.artifact_id == candidate.artifact_id
                or record.artifact_ref.uri == candidate.artifact_ref.uri
                or record.representation_ref.reference_id == candidate.representation_ref.reference_id
            )]
            if matches:
                prior = matches[0]
                if (
                    prior.representation_ref == candidate.representation_ref
                    and prior.receipt_ref == candidate.receipt_ref
                    and prior.expires_at == candidate.expires_at
                ):
                    results.append(prior)
                    continue
                raise ValueError(f"conflicting catalog registration: {candidate.artifact_id}")
            existing.append(candidate)
            results.append(candidate)
        self._assert_unique(existing)
        if tuple(existing) != self._records:
            self._write(tuple(existing))
        return tuple(results)

    def _verify_receipt(self, receipt_ref: ArtifactRef, references: tuple[RepresentationRef, ...]) -> None:
        """Verify bytes and ensure the receipt actually attests each reference."""
        path = self.resolver.resolve(receipt_ref)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("registration receipt is not valid JSON") from exc
        if not isinstance(payload, Mapping) or payload.get("status") != "complete":
            raise ValueError("registration receipt is incomplete")
        if not isinstance(payload.get("scene_id"), str) or not isinstance(payload.get("run_id"), str):
            raise ValueError("registration receipt lacks scene/run identity")
        rows = payload.get("representations")
        if not isinstance(rows, list):
            raise ValueError("registration receipt lacks representation rows")
        try:
            attested = [
                _representation_from_dict(row["reference"])
                for row in rows
                if isinstance(row, Mapping) and isinstance(row.get("reference"), Mapping)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("registration receipt contains a malformed representation") from exc
        for reference in references:
            if reference.scene_id != payload["scene_id"] or reference.artifact is None or reference.artifact.run_reference != payload["run_id"]:
                raise ValueError("registration reference conflicts with receipt scene/run")
            if reference not in attested:
                raise ValueError("registration reference is not attested by receipt")

    def _expire_due_records(self) -> None:
        now = _utc_now()
        changed = False
        records = []
        for record in self._records:
            if record.status is LifecycleStatus.ACTIVE and record.expires_at is not None and _parse_timestamp(record.expires_at, "expires_at") <= now:
                record = replace(record, status=LifecycleStatus.EXPIRED)
                changed = True
            records.append(record)
        if changed:
            self._write(tuple(records))

    def lookup(
        self,
        *,
        scene_id: str | None = None,
        representation_type: RepresentationType | str | None = None,
        modality: str | None = None,
        run_id: str | None = None,
        artifact_id: str | None = None,
        artifact_uri: str | None = None,
        spatial_level: str | None = None,
        active_only: bool = True,
    ) -> tuple[CatalogRecord, ...]:
        self._expire_due_records()
        kind = RepresentationType(representation_type) if representation_type is not None else None
        modality = modality.lower() if modality is not None else None
        spatial_level = spatial_level.lower() if spatial_level is not None else None
        return tuple(record for record in self._records if (
            (not active_only or record.status is LifecycleStatus.ACTIVE)
            and (scene_id is None or record.scene_id == scene_id)
            and (kind is None or record.representation_ref.representation_type is kind)
            and (modality is None or record.representation_ref.modality == modality)
            and (run_id is None or record.run_id == run_id)
            and (artifact_id is None or record.artifact_id == artifact_id)
            and (artifact_uri is None or record.artifact_ref.uri == artifact_uri)
            and (spatial_level is None or (
                record.representation_ref.spatial_reference is not None
                and record.representation_ref.spatial_reference.level == spatial_level
            ))
        ))

    def list_representations(self, scene_id: str, *, run_id: str | None = None) -> tuple[RepresentationRef, ...]:
        return tuple(record.representation_ref for record in self.lookup(scene_id=scene_id, run_id=run_id))

    def validate(self, record_or_artifact_id: CatalogRecord | str) -> ValidationResult:
        if isinstance(record_or_artifact_id, CatalogRecord):
            artifact_id = record_or_artifact_id.artifact_id
        elif isinstance(record_or_artifact_id, str):
            artifact_id = record_or_artifact_id
        else:
            raise TypeError("validation requires a CatalogRecord or artifact_id")
        matches = [record for record in self._records if record.artifact_id == artifact_id]
        if not matches:
            raise LookupError(f"artifact is not cataloged: {artifact_id}")
        record = matches[0]
        self._expire_due_records()
        record = next(item for item in self._records if item.artifact_id == artifact_id)
        if record.status is LifecycleStatus.EXPIRED:
            return ValidationResult(record, False, "artifact retention deadline has expired")
        if record.status is LifecycleStatus.INVALID:
            return ValidationResult(record, False, record.invalid_reason)
        try:
            self.resolver.load_numpy(record.representation_ref)
        except (FileNotFoundError, LookupError, OSError, TypeError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            invalid = replace(record, status=LifecycleStatus.INVALID, invalid_reason=reason)
            self._write(tuple(invalid if item.artifact_id == artifact_id else item for item in self._records))
            return ValidationResult(invalid, False, reason)
        return ValidationResult(record, True)

    def materialize(self, record_or_artifact_id: CatalogRecord | str):
        result = self.validate(record_or_artifact_id)
        if not result.valid:
            raise ValueError(f"catalog artifact is not reusable: {result.reason}")
        return self.resolver.load_numpy(result.record.representation_ref)

    def attach_to_scene(self, scene: SceneBundle, *, run_id: str | None = None) -> SceneBundle:
        """Return a new SceneBundle whose RepresentationSet includes discoveries."""
        if not isinstance(scene, SceneBundle):
            raise TypeError("scene must be a SceneBundle")
        discovered = self.list_representations(scene.scene_id, run_id=run_id or scene.analysis_id)
        by_id = {reference.reference_id: reference for reference in scene.representations.references}
        for reference in discovered:
            by_id.setdefault(reference.reference_id, reference)
        return replace(scene, representations=RepresentationSet(
            physical_features=scene.representations.physical_features,
            optical=scene.representations.optical,
            sar=scene.representations.sar,
            joint=scene.representations.joint,
            croma_scene=scene.representations.croma_scene,
            spatial_tokens=scene.representations.spatial_tokens,
            regions=scene.representations.regions,
            references=tuple(by_id.values()),
        ))
