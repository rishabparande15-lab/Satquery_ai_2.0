"""Strict BigEarthNet/CROMA foundation adapted from raviasha/Sat_Query.

This module is deliberately downstream-neutral: it validates and represents data for
SatQuery AI's advanced pipeline without importing the reference orchestration/UI.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

OPTICAL_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
SAR_BANDS = ("VV", "VH")
NORMALIZATION_PROFILE = "croma_readme_patch_8bit_v1"
TOKEN_GRID = (15, 15)
FEATURE_DIMENSION = 768

CLASSES = (
    ("Urban fabric", (111, 112)),
    ("Industrial or commercial units", (121,)),
    ("Arable land", (211, 212, 213)),
    ("Permanent crops", (221, 222, 223, 241)),
    ("Pastures", (231,)),
    ("Complex cultivation patterns", (242,)),
    ("Land principally occupied by agriculture, with significant areas of natural vegetation", (243,)),
    ("Agro-forestry areas", (244,)),
    ("Broad-leaved forest", (311,)),
    ("Coniferous forest", (312,)),
    ("Mixed forest", (313,)),
    ("Natural grassland and sparsely vegetated areas", (321, 333)),
    ("Moors, heathland and sclerophyllous vegetation", (322, 323)),
    ("Transitional woodland, shrub", (324,)),
    ("Beaches, dunes, sands", (331,)),
    ("Inland wetlands", (411, 412)),
    ("Coastal wetlands", (421, 422)),
    ("Inland waters", (511, 512)),
    ("Marine waters", (521, 522, 523)),
)
UNLABELED_CODES = (0, 999, 122, 123, 124, 131, 132, 133, 141, 142, 332, 334, 335, 423)
FEATURE_KEYS = ("optical_encodings", "SAR_encodings", "joint_encodings", "optical_GAP", "SAR_GAP", "joint_GAP")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SampleIdentity:
    patch_id: str
    s1_name: str
    s2_name: str
    split: str
    acquisition_metadata: dict[str, Any]
    crs: str
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    source_provenance: dict[str, Any]

    def validate(self) -> None:
        if not all(isinstance(v, str) and v.strip() for v in (self.patch_id, self.s1_name, self.s2_name)):
            raise ValueError("patch, Sentinel-1, and Sentinel-2 identities are required")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not self.crs or len(self.bounds) != 4 or len(self.resolution) != 2:
            raise ValueError("CRS, bounds, and resolution metadata are required")
        if not all(np.isfinite(self.bounds)) or not all(np.isfinite(self.resolution)) or not all(v > 0 for v in self.resolution):
            raise ValueError("bounds and resolution must be finite; resolution must be positive")
        if not self.acquisition_metadata or not self.source_provenance:
            raise ValueError("acquisition metadata and source provenance are required")


@dataclass(frozen=True)
class PreparedBatch:
    optical: torch.Tensor
    sar: torch.Tensor
    identities: tuple[SampleIdentity, ...]
    normalization: dict[str, Any]


def normalize_patch(image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reference profile: per-channel mean +/- 2 sample std, uint8, /255."""
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[1:] != (120, 120):
        raise ValueError("Expected a (channels,120,120) tensor")
    image = image.detach().cpu().to(torch.float32)
    if not torch.isfinite(image).all():
        raise ValueError("Normalization requires finite pixels")
    channels, means, stds, constants = [], [], [], []
    for index, channel in enumerate(image):
        if torch.equal(channel.min(), channel.max()):
            mean, std = channel[0, 0], channel.new_tensor(0)
        else:
            mean, std = channel.mean(), channel.std(correction=1)
        if std == 0:
            values = torch.full_like(channel, 127.5)
            constants.append(index)
        else:
            lo, hi = mean - 2 * std, mean + 2 * std
            values = ((channel - lo) / (hi - lo) * 255).clamp(0, 255)
        channels.append(values.to(torch.uint8).float() / 255)
        means.append(float(mean)); stds.append(float(std))
    return torch.stack(channels), {"profile": NORMALIZATION_PROFILE, "mean": means, "sample_std": stds, "constant_channels": constants}


def prepare_batch(optical: torch.Tensor, sar: torch.Tensor, identities: Iterable[SampleIdentity]) -> PreparedBatch:
    if optical.ndim != 4 or tuple(optical.shape[1:]) != (12, 120, 120):
        raise ValueError("Optical input must be [N,12,120,120] in canonical band order")
    if sar.ndim != 4 or tuple(sar.shape[1:]) != (2, 120, 120):
        raise ValueError("SAR input must be [N,2,120,120] in VV,VH order")
    identities = tuple(identities)
    if len(optical) != len(sar) or len(optical) != len(identities) or not len(optical):
        raise ValueError("Optical, SAR, and identity batch lengths must match and be nonempty")
    for identity in identities: identity.validate()
    if len({x.patch_id for x in identities}) != len(identities):
        raise ValueError("Duplicate patch identity in batch")
    normalized_o, normalized_s, metadata = [], [], []
    for o, s in zip(optical, sar):
        no, mo = normalize_patch(o); ns, ms = normalize_patch(s)
        normalized_o.append(no); normalized_s.append(ns); metadata.append({"optical": mo, "sar": ms})
    return PreparedBatch(torch.stack(normalized_o), torch.stack(normalized_s), identities,
                         {"profile": NORMALIZATION_PROFILE, "samples": metadata,
                          "optical_band_order": list(OPTICAL_BANDS), "sar_band_order": list(SAR_BANDS),
                          "reference_grid": "B02", "resampling": "bilinear after float32 conversion"})


def validate_croma_features(features: dict[str, torch.Tensor], sample_count: int) -> None:
    if set(features) != set(FEATURE_KEYS):
        raise ValueError(f"CROMA must return exactly {FEATURE_KEYS}")
    for key, value in features.items():
        expected = (sample_count, 225, 768) if key.endswith("encodings") else (sample_count, 768)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid CROMA feature {key}; expected finite float32 {expected}")


class FrozenCromaFoundation:
    def __init__(self, model: nn.Module, provenance: dict[str, Any]):
        if not provenance.get("checkpoint") or not provenance.get("checkpoint_sha256"):
            raise ValueError("CROMA checkpoint path and SHA-256 provenance are required")
        self.model = model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.provenance = dict(provenance)

    @torch.inference_mode()
    def __call__(self, batch: PreparedBatch) -> dict[str, torch.Tensor]:
        result = self.model(optical_images=batch.optical, SAR_images=batch.sar)
        result = {key: value.detach().cpu().to(torch.float32) for key, value in result.items()}
        validate_croma_features(result, len(batch.identities))
        return result


def patch_label_targets(reference_maps: torch.Tensor) -> dict[str, torch.Tensor]:
    if not isinstance(reference_maps, torch.Tensor) or reference_maps.ndim != 3 or tuple(reference_maps.shape[1:]) != (120, 120) or not len(reference_maps) or reference_maps.dtype.is_floating_point:
        raise ValueError("Reference maps must be nonempty integer [N,120,120] tensors")
    maps = reference_maps.detach().cpu().to(torch.int64)
    known = torch.tensor([c for _, codes in CLASSES for c in codes] + list(UNLABELED_CODES))
    unexpected = maps.unique()[~torch.isin(maps.unique(), known)]
    if len(unexpected): raise ValueError(f"Unknown reference-map codes: {unexpected.tolist()}")
    lookup = torch.full((1000,), 19, dtype=torch.int64)
    for index, (_, codes) in enumerate(CLASSES): lookup[list(codes)] = index
    blocks = lookup[maps].reshape(-1, 15, 8, 15, 8).permute(0, 1, 3, 2, 4).reshape(-1, 225, 64)
    histogram = torch.zeros(len(maps), 225, 20, dtype=torch.int64)
    histogram.scatter_add_(2, blocks, torch.ones_like(blocks))
    counts, unlabeled = histogram[..., :19].contiguous(), histogram[..., 19].contiguous()
    return {"class_counts": counts, "class_fractions": counts.float() / 64,
            "unlabeled_counts": unlabeled, "unlabeled_fraction": unlabeled.float() / 64,
            "fully_labeled_mask": unlabeled == 0, "has_labels_mask": unlabeled < 64}


class ReferenceCoverageHead(nn.Module):
    """REFERENCE BASELINE only: frozen-CROMA 768-vectors to 19 coverage fractions."""
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(768, 19)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dtype != torch.float32 or features.ndim < 2 or features.shape[-1] != 768 or not torch.isfinite(features).all():
            raise ValueError("Features must be finite float32 [...,768]")
        return self.linear(features)
    def fractions(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).softmax(-1)


def validate_split_integrity(splits: dict[str, Iterable[str]]) -> None:
    if set(splits) != {"train", "validation", "test"}: raise ValueError("Exactly train/validation/test splits are required")
    groups = {key: list(value) for key, value in splits.items()}
    if any(not values for values in groups.values()): raise ValueError("Every split must be nonempty")
    if any(len(values) != len(set(values)) for values in groups.values()): raise ValueError("Duplicate identity within split")
    if any(set(groups[a]) & set(groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Image/annotation identity leakage across splits")


def coverage_metrics(predictions: np.ndarray, truth: np.ndarray, area_ids: Iterable[str], *, bootstrap_repeats: int = 500, seed: int = 17) -> dict[str, Any]:
    p, y, ids = np.asarray(predictions, dtype=np.float64), np.asarray(truth, dtype=np.float64), np.asarray(list(area_ids))
    if p.ndim != 2 or p.shape != y.shape or p.shape[1] != 19 or ids.shape != (len(p),) or not len(p): raise ValueError("Expected aligned [M,19] fractions and area IDs")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6) or not np.allclose(y.sum(1), 1, atol=1e-6): raise ValueError("Coverage rows must be finite nonnegative distributions")
    error = (p-y)*100; absolute = np.abs(error); unique = np.unique(ids); rng = np.random.default_rng(seed)
    by_area = {a: absolute[ids == a] for a in unique}; boot = []
    for _ in range(bootstrap_repeats):
        chosen = rng.choice(unique, size=len(unique), replace=True); boot.append(np.concatenate([by_area[a] for a in chosen]).mean())
    actual, predicted = y.argmax(1), p.argmax(1); untied = (y == y.max(1, keepdims=True)).sum(1) == 1
    confusion = np.zeros((19, 19), dtype=int); np.add.at(confusion, (actual[untied], predicted[untied]), 1)
    per_class = [{"index": i, "name": CLASSES[i][0], "mae_pp": float(absolute[:, i].mean()), "rmse_pp": float(np.sqrt(np.square(error[:, i]).mean())), "bias_pp": float(error[:, i].mean()), "support_tokens": int((y[:, i] > 0).sum())} for i in range(19)]
    return {"mae_pp": float(absolute.mean()), "rmse_pp": float(np.sqrt(np.square(error).mean())), "bias_pp": float(error.mean()), "per_class": per_class, "dominant_class_accuracy": float(np.trace(confusion)/untied.sum()) if untied.any() else None, "confusion_matrix": confusion.tolist(), "bootstrap_area_mae_95ci": np.quantile(boot, [.025,.975]).tolist() if len(unique)>1 else None, "bootstrap": {"unit":"whole image area","repeats":bootstrap_repeats,"seed":seed}}


def write_experiment_report(path: Path, *, metrics: dict[str, Any], configuration: dict[str, Any], identities: Iterable[SampleIdentity], model: dict[str, Any], feature_contract: dict[str, Any]) -> Path:
    identities = tuple(identities); [item.validate() for item in identities]
    payload = {"format_version": 1, "phase": "phase1_foundation", "status": "complete", "metrics": metrics, "configuration": configuration, "dataset": [asdict(item) for item in identities], "model": model, "feature_contract": feature_contract}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")
    receipt = {"report": path.name, "sha256": sha256(path), "sample_ids": [x.patch_id for x in identities], "complete": True}
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    return path


def link_annotations(images: Iterable[SampleIdentity], rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservatively link untouched Q&A/caption/region records by both Sentinel IDs."""
    by_key = {(x.s2_name, x.s1_name): x for x in images}; result=[]
    for row in rows:
        key = (row.get("patch_id"), row.get("s1_name")); image = by_key.get(key)
        if image is None: continue
        if row.get("type") not in {"binary", "mcq", "captioning", "bounding box", "region"}: raise ValueError("Unknown annotation type")
        if not str(row.get("input", "")).strip() or not str(row.get("output", "")).strip(): raise ValueError("Empty annotation record")
        linked = dict(row); linked.update(image_split=image.split, patch_identity=image.patch_id,
            training_eligible=image.split == "train" and row.get("split") == "train",
            semantics_validated=False, feature_token_labels=False)
        result.append(linked)
    return result
