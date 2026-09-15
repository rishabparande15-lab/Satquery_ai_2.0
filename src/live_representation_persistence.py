"""Opt-in, receipt-backed persistence for already-computed live representations."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import torch

from .architecture_contracts import (
    ArtifactRef,
    ArtifactType,
    RepresentationRef,
    RepresentationType,
    SpatialReference,
)


class PersistableRepresentation(str, Enum):
    OPTICAL_CROMA_TOKENS = "optical_croma_tokens"
    SAR_CROMA_TOKENS = "sar_croma_tokens"
    JOINT_CROMA_TOKENS = "joint_croma_tokens"
    OPTICAL_CROMA_GAP = "optical_croma_gap"
    SAR_CROMA_GAP = "sar_croma_gap"
    JOINT_CROMA_GAP = "joint_croma_gap"
    RAW_OPTICAL = "raw_optical"
    RAW_SAR = "raw_sar"
    PHYSICAL_FEATURES = "physical_features"
    POOLED_CROMA = "pooled_croma"
    HYBRID = "hybrid"


CROMA_REPRESENTATIONS = frozenset({
    PersistableRepresentation.OPTICAL_CROMA_TOKENS,
    PersistableRepresentation.SAR_CROMA_TOKENS,
    PersistableRepresentation.JOINT_CROMA_TOKENS,
    PersistableRepresentation.OPTICAL_CROMA_GAP,
    PersistableRepresentation.SAR_CROMA_GAP,
    PersistableRepresentation.JOINT_CROMA_GAP,
})


@dataclass(frozen=True)
class RepresentationPersistencePolicy:
    """Explicit bounded policy; the default instance cannot write anything."""

    enabled: bool = False
    requested: frozenset[PersistableRepresentation] = frozenset()
    storage_root: Path | None = None
    namespace: str | None = None
    checksum_required: bool = True
    max_total_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.checksum_required) is not bool:
            raise ValueError("enabled and checksum_required must be boolean")
        try:
            selected = frozenset(PersistableRepresentation(value) for value in self.requested)
        except ValueError as exc:
            raise ValueError("requested contains an unsupported representation") from exc
        object.__setattr__(self, "requested", selected)
        if type(self.max_total_bytes) is not int or self.max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be a positive integer")
        if self.enabled:
            if not selected:
                raise ValueError("enabled persistence requires an explicit non-empty selection")
            if self.storage_root is None:
                raise ValueError("enabled persistence requires storage_root")
            if not self.namespace or not re.fullmatch(r"[A-Za-z0-9._-]+", self.namespace):
                raise ValueError("enabled persistence requires a simple logical namespace")
            if not self.checksum_required:
                raise ValueError("live representation persistence requires SHA-256 checksums")

    def selects(self, representation: PersistableRepresentation) -> bool:
        return self.enabled and representation in self.requested


@dataclass(frozen=True)
class PersistenceOutcome:
    references: tuple[RepresentationRef, ...]
    receipt: ArtifactRef
    writes: int
    reused: int
    total_bytes: int


@dataclass(frozen=True)
class _Descriptor:
    representation_type: RepresentationType
    modality: str
    spatial_level: str
    producer: str
    filename: str


_DESCRIPTORS = {
    PersistableRepresentation.OPTICAL_CROMA_TOKENS: _Descriptor(RepresentationType.OPTICAL_CROMA, "optical", "token", "CROMAAdapter", "croma_optical_encodings.npy"),
    PersistableRepresentation.SAR_CROMA_TOKENS: _Descriptor(RepresentationType.SAR_CROMA, "sar", "token", "CROMAAdapter", "croma_SAR_encodings.npy"),
    PersistableRepresentation.JOINT_CROMA_TOKENS: _Descriptor(RepresentationType.JOINT_CROMA, "optical_sar", "token", "CROMAAdapter", "croma_joint_encodings.npy"),
    PersistableRepresentation.OPTICAL_CROMA_GAP: _Descriptor(RepresentationType.OPTICAL_CROMA, "optical", "scene", "CROMAAdapter", "croma_optical_GAP.npy"),
    PersistableRepresentation.SAR_CROMA_GAP: _Descriptor(RepresentationType.SAR_CROMA, "sar", "scene", "CROMAAdapter", "croma_SAR_GAP.npy"),
    PersistableRepresentation.JOINT_CROMA_GAP: _Descriptor(RepresentationType.JOINT_CROMA, "optical_sar", "scene", "CROMAAdapter", "croma_joint_GAP.npy"),
    PersistableRepresentation.RAW_OPTICAL: _Descriptor(RepresentationType.RAW_OPTICAL, "optical", "pixel", "raster_inputs.assemble", "raw_optical.npy"),
    PersistableRepresentation.RAW_SAR: _Descriptor(RepresentationType.RAW_SAR, "sar", "pixel", "raster_inputs.assemble", "raw_sar.npy"),
    PersistableRepresentation.PHYSICAL_FEATURES: _Descriptor(RepresentationType.PHYSICAL_FEATURES, "optical_sar", "feature", "modality_features.extract", "physical_features.npy"),
    PersistableRepresentation.POOLED_CROMA: _Descriptor(RepresentationType.CROMA_SCENE, "optical_sar", "scene", "analysis_engine._deep_features", "pooled_croma_features.npy"),
    PersistableRepresentation.HYBRID: _Descriptor(RepresentationType.HYBRID, "optical_sar", "scene", "HybridFusion", "hybrid_features.npy"),
}

_CROMA_KEYS = {
    "optical_encodings": PersistableRepresentation.OPTICAL_CROMA_TOKENS,
    "SAR_encodings": PersistableRepresentation.SAR_CROMA_TOKENS,
    "joint_encodings": PersistableRepresentation.JOINT_CROMA_TOKENS,
    "optical_GAP": PersistableRepresentation.OPTICAL_CROMA_GAP,
    "SAR_GAP": PersistableRepresentation.SAR_CROMA_GAP,
    "joint_GAP": PersistableRepresentation.JOINT_CROMA_GAP,
}


class RepresentationPersistenceSession:
    """Run-local collector and immutable writer for an explicit policy."""

    def __init__(self, policy: RepresentationPersistencePolicy | None = None) -> None:
        self.policy = policy or RepresentationPersistencePolicy()
        self._captured: dict[PersistableRepresentation, tuple[np.ndarray, dict[str, str | None]]] = {}
        self._outcome: PersistenceOutcome | None = None

    @property
    def outcome(self) -> PersistenceOutcome | None:
        return self._outcome

    def captured(self, representation: PersistableRepresentation) -> np.ndarray | None:
        """Return a defensive copy of selected run-local material for identity checks."""
        item = self._captured.get(PersistableRepresentation(representation))
        return None if item is None else item[0].copy()

    def capture(
        self,
        representation: PersistableRepresentation,
        value: np.ndarray | torch.Tensor,
        *,
        modality: str | None = None,
        producer_version: str | None = None,
        preprocessing_version: str | None = None,
    ) -> None:
        representation = PersistableRepresentation(representation)
        if not self.policy.selects(representation):
            return
        if representation in self._captured:
            raise ValueError(f"representation captured twice: {representation.value}")
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            array = value
        else:
            raise TypeError("persisted representation material must be a NumPy array or tensor")
        material = np.ascontiguousarray(array).copy()
        if material.dtype.hasobject:
            raise ValueError("object arrays cannot be persisted")
        self._captured[representation] = (material, {
            "modality": modality,
            "producer_version": producer_version,
            "preprocessing_version": preprocessing_version,
        })

    def capture_croma(
        self,
        outputs: Mapping[str, torch.Tensor],
        *,
        producer_version: str | None,
        preprocessing_version: str,
    ) -> None:
        for key, representation in _CROMA_KEYS.items():
            if key in outputs:
                self.capture(
                    representation,
                    outputs[key][0],
                    producer_version=producer_version,
                    preprocessing_version=preprocessing_version,
                )

    def finalize(
        self,
        *,
        scene_id: str,
        run_id: str,
        crs: str | None,
        provenance: Mapping[str, Any],
    ) -> PersistenceOutcome | None:
        if not self.policy.enabled:
            return None
        if self._outcome is not None:
            return self._outcome
        missing = self.policy.requested.difference(self._captured)
        if missing:
            names = ", ".join(sorted(value.value for value in missing))
            raise ValueError(f"requested representations were not produced: {names}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", scene_id) or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("scene_id and run_id must be safe logical identifiers")

        serialized = {}
        total_bytes = 0
        for representation in sorted(self._captured, key=lambda value: value.value):
            array, _ = self._captured[representation]
            stream = BytesIO()
            np.save(stream, array, allow_pickle=False)
            content = stream.getvalue()
            serialized[representation] = content
            total_bytes += len(content)
        if total_bytes > self.policy.max_total_bytes:
            raise ValueError(f"selected representations exceed max_total_bytes: {total_bytes}")

        root = Path(self.policy.storage_root).resolve()
        collection = root / scene_id / run_id
        collection.mkdir(parents=True, exist_ok=True)
        if not collection.resolve().is_relative_to(root):
            raise ValueError("persistence collection escapes storage_root")
        namespace = str(self.policy.namespace)
        receipt_uri = f"artifact://{namespace}/{scene_id}/{run_id}/receipt.json"
        references = []
        writes = reused = 0
        receipt_rows = []
        for representation in sorted(self._captured, key=lambda value: value.value):
            array, metadata = self._captured[representation]
            descriptor = _DESCRIPTORS[representation]
            content = serialized[representation]
            checksum = hashlib.sha256(content).hexdigest()
            path = collection / descriptor.filename
            wrote = self._write_immutable(path, content, checksum)
            self._verify_numpy(path, array, checksum, len(content))
            writes += int(wrote)
            reused += int(not wrote)
            uri = f"artifact://{namespace}/{scene_id}/{run_id}/{descriptor.filename}"
            artifact_id = f"{scene_id}:{run_id}:{representation.value}:{checksum[:16]}"
            reference_id = f"representation:{scene_id}:{run_id}:{representation.value}:{checksum[:16]}"
            artifact = ArtifactRef(
                artifact_id=artifact_id,
                artifact_type=ArtifactType.NUMPY_ARRAY,
                uri=uri,
                checksum_sha256=checksum,
                size_bytes=len(content),
                media_type="application/x-npy",
                format="npy",
                producer=descriptor.producer,
                producer_version=metadata["producer_version"],
                run_reference=run_id,
                provenance_reference=receipt_uri,
            )
            spatial = SpatialReference(
                level=descriptor.spatial_level,
                crs=crs if descriptor.spatial_level in {"pixel", "token"} else None,
                token_grid=(15, 15) if descriptor.spatial_level == "token" else None,
                mapping_reference="spatial_evidence.scene.token_grid" if descriptor.spatial_level == "token" else None,
                mapping_version="north_up_row_major_120_to_15_v1" if descriptor.spatial_level == "token" else None,
            )
            reference = RepresentationRef(
                reference_id=reference_id,
                scene_id=scene_id,
                representation_type=descriptor.representation_type,
                modality=metadata["modality"] or descriptor.modality,
                shape=tuple(array.shape),
                dtype=str(array.dtype),
                spatial_reference=spatial,
                producer=descriptor.producer,
                producer_version=metadata["producer_version"],
                preprocessing_version=metadata["preprocessing_version"],
                artifact=artifact,
                provenance_reference=receipt_uri,
                status="available",
            )
            references.append(reference)
            receipt_rows.append({
                "selection": representation.value,
                "reference": reference.to_dict(),
            })

        receipt_payload = {
            "receipt_version": 1,
            "status": "complete",
            "scene_id": scene_id,
            "run_id": run_id,
            "namespace": namespace,
            "representations": receipt_rows,
            "provenance": dict(provenance),
        }
        receipt_content = (json.dumps(receipt_payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        receipt_path = collection / "receipt.json"
        receipt_wrote = self._write_immutable(receipt_path, receipt_content, hashlib.sha256(receipt_content).hexdigest())
        writes += int(receipt_wrote)
        reused += int(not receipt_wrote)
        receipt_ref = ArtifactRef(
            artifact_id=f"{scene_id}:{run_id}:receipt:{hashlib.sha256(receipt_content).hexdigest()[:16]}",
            artifact_type=ArtifactType.PROVENANCE_RECEIPT,
            uri=receipt_uri,
            checksum_sha256=hashlib.sha256(receipt_content).hexdigest(),
            size_bytes=len(receipt_content),
            media_type="application/json",
            format="json",
            producer="RepresentationPersistenceSession",
            producer_version="live_representation_persistence_v1",
            run_reference=run_id,
        )
        # Registration is deliberately last: only fully written and verified
        # representations, plus their verified receipt, cross into the catalog.
        # The files remain immutable if catalog registration fails; callers get
        # an explicit failure and no successful PersistenceOutcome.
        try:
            from .receipt_catalog import ReceiptCatalog
            from .representation_artifacts import ArtifactResolver

            catalog = ReceiptCatalog(root / "receipt_catalog.json", ArtifactResolver({namespace: root}))
            catalog.register_many(references, receipt_ref)
        except Exception as exc:
            raise RuntimeError(f"artifacts persisted but catalog registration failed: {exc}") from exc
        self._outcome = PersistenceOutcome(tuple(references), receipt_ref, writes, reused, total_bytes + len(receipt_content))
        return self._outcome

    @staticmethod
    def _write_immutable(path: Path, content: bytes, checksum: str) -> bool:
        try:
            with path.open("xb") as stream:
                stream.write(content)
            return True
        except FileExistsError:
            existing = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing != checksum:
                raise FileExistsError(f"immutable artifact conflict: {path.name}") from None
            return False

    @staticmethod
    def _verify_numpy(path: Path, source: np.ndarray, checksum: str, size_bytes: int) -> None:
        if path.stat().st_size != size_bytes:
            raise ValueError(f"persisted artifact size mismatch: {path.name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"persisted artifact checksum mismatch: {path.name}")
        loaded = np.load(path, allow_pickle=False)
        if loaded.shape != source.shape or loaded.dtype != source.dtype:
            raise ValueError(f"persisted artifact metadata mismatch: {path.name}")
        if not np.array_equal(loaded, source, equal_nan=True):
            raise ValueError(f"persisted artifact content mismatch: {path.name}")
