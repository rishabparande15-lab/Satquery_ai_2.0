"""Logical artifact references and a minimal, explicit materialization boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np

from .architecture_contracts import (
    ArtifactRef,
    ArtifactType,
    RepresentationRef,
    RepresentationType,
    SpatialReference,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactResolver:
    """Resolve explicitly configured artifact namespaces; no global storage state."""

    def __init__(self, namespaces: Mapping[str, Path]) -> None:
        self._namespaces = {
            name: Path(root).resolve()
            for name, root in namespaces.items()
        }
        if any(not name or "/" in name or "\\" in name for name in self._namespaces):
            raise ValueError("artifact namespace names must be simple identifiers")

    def resolve(self, artifact: ArtifactRef, *, verify: bool = True) -> Path:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        parsed = urlparse(artifact.uri)
        root = self._namespaces.get(parsed.netloc)
        if root is None:
            raise LookupError(f"unconfigured artifact namespace: {parsed.netloc}")
        parts = tuple(part for part in parsed.path.split("/") if part)
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError("artifact URI contains an invalid logical path")
        path = root.joinpath(*parts).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact URI escapes its configured namespace")
        if not path.is_file():
            raise FileNotFoundError(f"artifact is not materialized: {artifact.uri}")
        if verify:
            if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
                raise ValueError(f"artifact size mismatch: {artifact.artifact_id}")
            if artifact.checksum_sha256 is not None and file_sha256(path) != artifact.checksum_sha256:
                raise ValueError(f"artifact checksum mismatch: {artifact.artifact_id}")
        return path

    def load_numpy(self, reference: RepresentationRef) -> np.ndarray:
        """Load one verified NumPy representation and validate its contract metadata."""
        if not isinstance(reference, RepresentationRef):
            raise TypeError("reference must be a RepresentationRef")
        if not reference.addressable or reference.artifact is None:
            raise ValueError("representation is not addressable")
        if reference.artifact.artifact_type != ArtifactType.NUMPY_ARRAY:
            raise ValueError("representation artifact is not a NumPy array")
        array = np.load(self.resolve(reference.artifact), allow_pickle=False)
        if reference.shape is not None and tuple(array.shape) != reference.shape:
            raise ValueError("materialized representation shape does not match its reference")
        if reference.dtype is not None and str(array.dtype) != reference.dtype:
            raise ValueError("materialized representation dtype does not match its reference")
        return array


_ARRAY_TYPES = {
    "raw_optical": (RepresentationType.RAW_OPTICAL, "optical", "pixel", "raster_inputs.assemble"),
    "raw_sar": (RepresentationType.RAW_SAR, "sar", "pixel", "raster_inputs.assemble"),
    "physical_features": (RepresentationType.PHYSICAL_FEATURES, "optical_sar", "feature", "modality_features.extract"),
    "pooled_croma_features": (RepresentationType.CROMA_SCENE, "optical_sar", "scene", "phase2_integration.adapt_phase1_to_phase2"),
    "hybrid_features": (RepresentationType.HYBRID, "optical_sar", "scene", "HybridFusion"),
    "croma_optical_encodings": (RepresentationType.OPTICAL_CROMA, "optical", "token", "CROMAAdapter"),
    "croma_SAR_encodings": (RepresentationType.SAR_CROMA, "sar", "token", "CROMAAdapter"),
    "croma_joint_encodings": (RepresentationType.JOINT_CROMA, "optical_sar", "token", "CROMAAdapter"),
    "croma_optical_GAP": (RepresentationType.OPTICAL_CROMA, "optical", "scene", "CROMAAdapter"),
    "croma_SAR_GAP": (RepresentationType.SAR_CROMA, "sar", "scene", "CROMAAdapter"),
    "croma_joint_GAP": (RepresentationType.JOINT_CROMA, "optical_sar", "scene", "CROMAAdapter"),
}


def representation_refs_from_receipt(
    receipt: Mapping[str, Any],
    *,
    namespace: str,
    collection: str | None = None,
) -> tuple[RepresentationRef, ...]:
    """Create verified-metadata references without embedding the receipt's local path."""
    if receipt.get("status") != "complete" or not receipt.get("sample_id"):
        raise ValueError("artifact receipt must be complete and identify a sample")
    scene_id = str(receipt["sample_id"])
    logical_collection = collection or scene_id
    if not logical_collection or any(character in logical_collection for character in "/\\") or logical_collection in {".", ".."}:
        raise ValueError("collection must be a simple logical identifier")
    arrays = receipt.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("artifact receipt arrays must be a mapping")
    provenance = receipt.get("provenance") or {}
    acquisition = provenance.get("acquisition") or {}
    phase1 = provenance.get("phase1") or {}
    phase2 = provenance.get("phase2") or {}
    phase2_provenance = phase2.get("provenance") or {}
    croma_provenance = phase2_provenance.get("croma") or {}
    physical_provenance = phase2_provenance.get("physical") or {}
    mapping_version = "north_up_row_major_120_to_15_v1"
    provenance_uri = f"artifact://{namespace}/{logical_collection}/receipt.json"
    references = []
    for name, metadata in arrays.items():
        if name not in _ARRAY_TYPES:
            raise ValueError(f"unsupported representation in receipt: {name}")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid artifact metadata for {name}")
        kind, modality, spatial_level, producer = _ARRAY_TYPES[name]
        checksum = metadata.get("sha256")
        filename = metadata.get("file")
        if not isinstance(checksum, str):
            raise ValueError(f"artifact checksum is unavailable for {name}")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"artifact filename is unavailable for {name}")
        if producer == "CROMAAdapter":
            producer_version = croma_provenance.get("checkpoint_sha256")
        elif producer == "modality_features.extract":
            producer_version = physical_provenance.get("schema")
        elif producer.startswith("phase2_integration"):
            producer_version = phase2.get("adapter_version")
        else:
            producer_version = None
        artifact = ArtifactRef(
            artifact_id=f"{scene_id}:{name}:{str(checksum)[:16]}",
            artifact_type=ArtifactType.NUMPY_ARRAY,
            uri=f"artifact://{namespace}/{logical_collection}/{filename}",
            checksum_sha256=checksum,
            format="npy",
            media_type="application/x-npy",
            producer=producer,
            producer_version=producer_version,
            run_reference=str((provenance.get("request") or {}).get("request_id") or "") or None,
            provenance_reference=provenance_uri,
        )
        spatial = SpatialReference(
            level=spatial_level,
            crs=acquisition.get("crs") if spatial_level in {"pixel", "token", "region", "evidence"} else None,
            token_grid=(15, 15) if spatial_level == "token" else None,
            mapping_reference=(
                f"artifact://spatial-evidence/{scene_id}/scene.json"
                if spatial_level == "token" else None
            ),
            mapping_version=mapping_version if spatial_level == "token" else None,
        )
        references.append(RepresentationRef(
            reference_id=f"representation:{scene_id}:{name}:{str(checksum)[:16]}",
            scene_id=scene_id,
            representation_type=kind,
            modality=modality,
            shape=tuple(metadata.get("shape") or ()),
            dtype=metadata.get("dtype"),
            spatial_reference=spatial,
            producer=producer,
            producer_version=producer_version,
            preprocessing_version=phase1.get("normalization_profile"),
            artifact=artifact,
            provenance_reference=provenance_uri,
            status="available",
        ))
    return tuple(references)


def artifact_ref_from_file(
    path: Path,
    *,
    artifact_id: str,
    artifact_type: ArtifactType,
    logical_uri: str,
    producer: str,
    producer_version: str | None = None,
    provenance_reference: str | None = None,
    media_type: str | None = None,
    format: str | None = None,
) -> ArtifactRef:
    """Describe an existing file while keeping its filesystem path outside the contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        uri=logical_uri,
        checksum_sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        format=format,
        producer=producer,
        producer_version=producer_version,
        provenance_reference=provenance_reference,
    )
