from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .croma_adapter import CROMAAdapter
from .dataset_loader import discover_samples, load_sample
from .gee_features import LocalRasterFeatureProvider
from .hybrid_fusion import pooled_croma_features
from .training_data import BIGEARTHNET_CLASSES, encode_labels, load_labels

LOGGER = logging.getLogger(__name__)


def _cache_valid(sample_root: Path, report: dict) -> bool:
    try:
        gee = np.load(sample_root / "gee_features.npy", allow_pickle=False)
        croma = np.load(sample_root / "croma_pooled.npy", allow_pickle=False)
        labels = np.load(sample_root / "labels.npy", allow_pickle=False)
        return (gee.ndim == 1 and croma.ndim == 1 and labels.ndim == 1 and np.isfinite(gee).all() and np.isfinite(croma).all() and np.isfinite(labels).all() and gee.size == report.get("gee_dimension") and croma.size == report.get("croma_dimension") and labels.size == report.get("label_dimension"))
    except (OSError, ValueError, EOFError):
        return False


def extract_feature_cache(dataset_root: Path, cache_root: Path, croma_source: Path, checkpoint: Path, device: str | None = None, resume: bool = True) -> dict:
    started = time.perf_counter()
    samples = discover_samples(dataset_root)
    labels = load_labels(dataset_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    provider = LocalRasterFeatureProvider()
    reports = []
    pending = []
    for sample in samples:
        sample_root = cache_root / sample.patch_id
        report_path = sample_root / "metadata.json"
        if resume and report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                if existing.get("status") == "complete" and _cache_valid(sample_root, existing):
                    reports.append(existing); continue
                LOGGER.warning("Invalid or corrupted cache detected for %s; re-extracting", sample.patch_id)
            except (OSError, json.JSONDecodeError):
                pass
        pending.append((sample, sample_root, report_path))
    adapter = CROMAAdapter(croma_source, checkpoint, device=device) if pending else None
    for sample, sample_root, report_path in pending:
        sample_root.mkdir(parents=True, exist_ok=True)
        try:
            item = load_sample(sample)
            gee, gee_report = provider.extract(item.raw_optical, item.raw_sar, item.metadata["optical_band_order"], item.metadata["sar_band_order"])
            outputs = adapter.infer(item.raw_optical, item.raw_sar)
            croma = pooled_croma_features(outputs)
            target = encode_labels(labels[sample.patch_id], BIGEARTHNET_CLASSES)
            if not np.isfinite(gee).all() or not np.isfinite(croma).all():
                raise ValueError("Non-finite cached feature")
            np.save(sample_root / "gee_features.npy", gee)
            np.save(sample_root / "croma_pooled.npy", croma)
            np.save(sample_root / "labels.npy", target)
            report = {"status": "complete", "sample_id": sample.patch_id, "gee_dimension": int(gee.size), "croma_dimension": int(croma.size), "label_dimension": int(target.size), "labels": labels[sample.patch_id], "gee_feature_names": gee_report["feature_names"], "device": str(adapter.device), "runtime_seconds": time.perf_counter() - started}
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            reports.append(report)
        except Exception as error:
            failure = {"status": "failed", "sample_id": sample.patch_id, "error": f"{type(error).__name__}: {error}"}
            report_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            LOGGER.exception("Feature extraction failed for %s", sample.patch_id)
            raise
    summary = {"dataset_root": str(dataset_root), "cache_root": str(cache_root), "samples_processed": len(reports), "reports": reports, "runtime_seconds": time.perf_counter() - started, "resume": resume}
    (cache_root / "cache_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
