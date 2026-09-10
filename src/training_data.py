from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BIGEARTHNET_CLASSES = [
    "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops",
    "Pastures", "Complex cultivation patterns", "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland",
    "Moors and heathland", "Sclerophyllous vegetation", "Transitional woodland/shrub", "Beaches, dunes, sands",
    "Inland wetlands", "Coastal wetlands", "Inland waters",
]


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    ratios: dict[str, float]
    class_names: list[str]
    splits: dict[str, list[str]]

    def to_dict(self) -> dict:
        return {"seed": self.seed, "ratios": self.ratios, "class_names": self.class_names, "splits": self.splits}


def encode_labels(labels: list[str] | tuple[str, ...] | np.ndarray, class_names: list[str]) -> np.ndarray:
    unknown = sorted(set(str(label) for label in labels) - set(class_names))
    if unknown:
        raise ValueError(f"Unknown labels not present in class schema: {unknown}")
    vector = np.zeros(len(class_names), dtype=np.float32)
    for label in labels:
        vector[class_names.index(str(label))] = 1.0
    if not vector.any():
        raise ValueError("Each sample must have at least one label")
    return vector


def load_labels(dataset_root: Path, class_names: list[str] | None = None) -> dict[str, list[str]]:
    metadata_path = dataset_root / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata parquet: {metadata_path}")
    import pandas as pd
    frame = pd.read_parquet(metadata_path)
    if "patch_id" not in frame or "labels" not in frame:
        raise ValueError("Metadata must contain patch_id and labels columns")
    result = {}
    for row in frame.itertuples(index=False):
        patch = str(row.patch_id).rsplit("_", 2)[-2] + "_" + str(row.patch_id).rsplit("_", 2)[-1]
        labels = [str(value) for value in row.labels]
        encode_labels(labels, class_names or BIGEARTHNET_CLASSES)
        result[patch] = labels
    return result


def make_splits(sample_ids: list[str], seed: int = 42, ratios: dict[str, float] | None = None) -> SplitManifest:
    ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if set(ratios) != {"train", "validation", "test"} or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Split ratios must contain train, validation, test and sum to 1")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs cannot be split")
    ids = list(sample_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    counts = {key: int(n * value) for key, value in ratios.items()}
    if n >= 3:
        if n < 10:
            counts = {"train": 1, "validation": 1, "test": 1}
        else:
            for key in ("train", "validation", "test"):
                if counts[key] == 0:
                    donor = max((name for name in counts if counts[name] > 1), key=lambda name: counts[name], default=None)
                    if donor is None:
                        raise ValueError("Unable to create non-empty train, validation, and test splits")
                    counts[donor] -= 1
                    counts[key] = 1
    while sum(counts.values()) < n:
        counts["train"] += 1
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    splits = {"train": ids[:train_end], "validation": ids[train_end:validation_end], "test": ids[validation_end:]}
    if set().union(*splits.values()) != set(ids) or any(set(a) & set(b) for a, b in ((splits["train"], splits["validation"]), (splits["train"], splits["test"]), (splits["validation"], splits["test"]))):
        raise RuntimeError("Split leakage detected")
    return SplitManifest(seed, ratios, BIGEARTHNET_CLASSES, splits)


def save_manifest(manifest: SplitManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> SplitManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(int(data["seed"]), dict(data["ratios"]), list(data["class_names"]), {key: list(value) for key, value in data["splits"].items()})
