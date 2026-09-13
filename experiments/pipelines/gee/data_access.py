from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.dataset_loader import Sample, discover_samples, load_sample
from src.gee_experiment_runtime import gee_status, select_mode


@dataclass
class LocalReference:
    sample: Sample
    prepared: object
    source: str = "local BigEarthNet reference; not GEE imagery"


def load_local_references(dataset_root: Path, sample_ids: tuple[str, ...]) -> list[LocalReference]:
    samples = {sample.patch_id: sample for sample in discover_samples(dataset_root)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples]
    if missing:
        raise FileNotFoundError(f"Missing local reference samples: {missing}")
    return [LocalReference(samples[sample_id], load_sample(samples[sample_id])) for sample_id in sample_ids]


def local_summary(reference: LocalReference) -> dict:
    item = reference.prepared
    return {
        "sample_id": reference.sample.patch_id,
        "source": reference.source,
        "dates": None,
        "temporal_metadata_available": False,
        "optical_bands": list(item.metadata["optical_band_order"]),
        "sar_bands": list(item.metadata["sar_band_order"]),
        "optical_shape": list(item.raw_optical.shape),
        "sar_shape": list(item.raw_sar.shape),
        "crs": item.metadata["crs"],
        "resolution": item.metadata["resolution"],
        "preprocessing": item.metadata,
        "summary_statistics": {
            "optical_min": float(np.nanmin(item.raw_optical)),
            "optical_max": float(np.nanmax(item.raw_optical)),
            "sar_min": float(np.nanmin(item.raw_sar)),
            "sar_max": float(np.nanmax(item.raw_sar)),
        },
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
