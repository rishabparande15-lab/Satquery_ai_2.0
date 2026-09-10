"""Typed Phase 1 to existing advanced-pipeline integration boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch

from .phase1_foundation import FEATURE_KEYS, OPTICAL_BANDS, SAR_BANDS, PreparedBatch, validate_croma_features


@dataclass(frozen=True)
class Phase2PipelineInput:
    identities: tuple[Any, ...]
    optical_encodings: torch.Tensor
    sar_encodings: torch.Tensor
    joint_encodings: torch.Tensor
    optical_gap: torch.Tensor
    sar_gap: torch.Tensor
    joint_gap: torch.Tensor
    physical_features: np.ndarray
    physical_feature_names: tuple[str, ...]
    pooled_croma_features: np.ndarray
    provenance: tuple[dict[str, Any], ...]
    missing_optional_features: tuple[str, ...] = ()


def _validate_physical(features: np.ndarray, reports: tuple[dict[str, Any], ...], identities: tuple[Any, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(features, dtype=np.float32)
    count = len(identities)
    if values.ndim != 2 or values.shape[0] != count or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError(f"Physical features must be finite [N,D], got {values.shape}")
    if len(reports) != count:
        raise ValueError("Physical reports must align with sample identities")
    names = tuple(reports[0].get("feature_names", ()))
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("Physical feature names must match the feature dimension and be unique")
    for index, report in enumerate(reports):
        if report.get("sample_id") is not None and report["sample_id"] != identities[index].patch_id:
            raise ValueError("Physical feature sample identity does not match Phase 1 identity")
        if tuple(report.get("feature_names", ())) != names or report.get("dimension") != values.shape[1]:
            raise ValueError("Physical feature schema changed across samples")
    return values, names


def adapt_phase1_output(
    batch: PreparedBatch,
    croma_features: dict[str, torch.Tensor],
    physical_features: np.ndarray,
    physical_reports: Iterable[dict[str, Any]],
    *,
    croma_provenance: dict[str, Any],
    experiment_configuration: dict[str, Any] | None = None,
    source_hashes: Iterable[dict[str, str]] | None = None,
    missing_optional_features: Iterable[str] = (),
) -> Phase2PipelineInput:
    """Validate and package one aligned batch for the unchanged fusion path."""
    if not isinstance(batch, PreparedBatch):
        raise TypeError("batch must be a Phase 1 PreparedBatch")
    if not croma_provenance.get("checkpoint") or not croma_provenance.get("checkpoint_sha256"):
        raise ValueError("CROMA checkpoint and SHA-256 provenance are required")
    identities = tuple(batch.identities)
    if len({identity.patch_id for identity in identities}) != len(identities):
        raise ValueError("Duplicate sample identity in Phase 2 input")
    splits = {identity.split for identity in identities}
    if len(splits) != 1:
        raise ValueError("A Phase 2 batch cannot mix train, validation, and test splits")
    validate_croma_features(croma_features, len(identities))
    reports = tuple(physical_reports)
    physical, names = _validate_physical(physical_features, reports, identities)
    hashes = tuple(source_hashes or ({ } for _ in identities))
    if len(hashes) != len(identities):
        raise ValueError("Source hashes must align with sample identities")
    pooled = torch.cat((croma_features["optical_GAP"], croma_features["SAR_GAP"], croma_features["joint_GAP"]), dim=-1).detach().cpu().numpy().astype(np.float32)
    if pooled.shape != (len(identities), 2304) or not np.isfinite(pooled).all():
        raise ValueError(f"Existing advanced pipeline requires pooled CROMA [N,2304], got {pooled.shape}")
    configuration = dict(experiment_configuration or {})
    provenance = []
    for index, identity in enumerate(identities):
        provenance.append({
            "identity": asdict(identity),
            "normalization": batch.normalization,
            "croma": {**croma_provenance, "feature_keys": list(FEATURE_KEYS), "spatial_shape": [225, 768], "pooled_shape": [768]},
            "physical": {"schema": reports[index].get("schema", "provider_reported"), "feature_names": list(names), "dimension": int(physical.shape[1])},
            "channel_order": {"optical": list(OPTICAL_BANDS), "sar": list(SAR_BANDS)},
            "experiment_configuration": configuration,
            "source_hashes": dict(hashes[index]),
        })
    return Phase2PipelineInput(
        identities=identities,
        optical_encodings=croma_features["optical_encodings"].detach().cpu(),
        sar_encodings=croma_features["SAR_encodings"].detach().cpu(),
        joint_encodings=croma_features["joint_encodings"].detach().cpu(),
        optical_gap=croma_features["optical_GAP"].detach().cpu(),
        sar_gap=croma_features["SAR_GAP"].detach().cpu(),
        joint_gap=croma_features["joint_GAP"].detach().cpu(),
        physical_features=physical,
        physical_feature_names=names,
        pooled_croma_features=pooled,
        provenance=tuple(provenance),
        missing_optional_features=tuple(missing_optional_features),
    )