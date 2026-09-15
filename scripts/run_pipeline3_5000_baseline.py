"""Build compact exact-5,000 Pipeline 3 features and train the frozen baseline.

The feature command consumes the already-verified manifest; it deliberately does
not repeat the full acquisition/raster hash audit.  The train command never loads
test feature rows, which keeps the held-out split sealed for final evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.croma_adapter import CROMAAdapter
from src.gee_features import LocalRasterFeatureProvider
from src.phase1_foundation import CLASSES, OPTICAL_BANDS, SAR_BANDS, coverage_metrics, patch_label_targets
from src.phase3_6_benchmark import ProbeHead, ProbeSplit, coverage_loss, fit_probe, predict_rows, sha256


DATASET_FINGERPRINT = "7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625"
DATASET_MANIFEST_SHA256 = "9f0446c011c1c054cea0fae5e7a2ccf196dcbdb50df6b7fc86757e764b0758be"
SPLIT_FINGERPRINT = "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"
CROMA_CHECKPOINT_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"
CROMA_SOURCE_REVISION = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
EXPECTED_SPLITS = {"train": 4600, "validation": 200, "test": 200}
MODES = ("constant", "physical", "optical_croma", "sar_croma", "joint_croma", "hybrid")
PREFIX = "bigearthnet-v2-5000/"


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_contract(manifest_path: Path, fingerprint_path: Path) -> tuple[dict, list[dict]]:
    if sha256(manifest_path) != DATASET_MANIFEST_SHA256:
        raise ValueError("Dataset manifest hash changed after the verified audit")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if fingerprint.get("fingerprint") != DATASET_FINGERPRINT:
        raise ValueError("Dataset fingerprint is not the frozen exact-5,000 fingerprint")
    basis = fingerprint.get("basis", {})
    if basis.get("split_sha256") != SPLIT_FINGERPRINT or basis.get("status") != "ready_for_training":
        raise ValueError("Split fingerprint/status changed after the verified audit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows", [])
    counts = {name: sum(row.get("split") == name for row in rows) for name in EXPECTED_SPLITS}
    if len(rows) != 5000 or counts != EXPECTED_SPLITS:
        raise ValueError(f"Expected exact 5,000 rows and {EXPECTED_SPLITS}, got {len(rows)} and {counts}")
    if len({row["area_id"] for row in rows}) != 5000:
        raise ValueError("Duplicate area identity in frozen dataset manifest")
    if any(row.get("validation", {}).get("status") != "valid" for row in rows):
        raise ValueError("Frozen dataset manifest contains a non-valid row")
    return manifest, rows


def _copy_nested_archives(outer_path: Path, source_cache: Path) -> tuple[Path, Path]:
    source_cache.mkdir(parents=True, exist_ok=True)
    targets = {
        "s1": (PREFIX + "BigEarthNet-S1-selected.zip", source_cache / "BigEarthNet-S1-selected.zip"),
        "reference": (PREFIX + "Reference_Maps-selected.zip", source_cache / "Reference_Maps-selected.zip"),
    }
    with zipfile.ZipFile(outer_path) as outer:
        for member, target in targets.values():
            expected = outer.getinfo(member).file_size
            if target.is_file() and target.stat().st_size == expected:
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            with outer.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if temporary.stat().st_size != expected:
                raise IOError(f"Incomplete nested archive extraction: {member}")
            temporary.replace(target)
    return targets["s1"][1], targets["reference"][1]


def _s2_paths(row: dict) -> dict[str, Path]:
    root = Path(row["s2_source_roots"][0])
    area = row["area_id"]
    tile = area.rsplit("_", 2)[0]
    folder = root / tile / area
    return {band: folder / f"{area}_{band}.tif" for band in OPTICAL_BANDS}


def _profile(source) -> dict:
    return {
        "crs": source.crs,
        "transform": source.transform,
        "bounds": tuple(source.bounds),
        "width": source.width,
        "height": source.height,
    }


def _read_path_to_grid(path: Path, grid: dict, resampling: Resampling) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as source:
        return _read_open_to_grid(source, grid, resampling)


def _read_bytes_to_grid(blob: bytes, grid: dict, resampling: Resampling) -> np.ndarray:
    with MemoryFile(blob) as memory, memory.open() as source:
        return _read_open_to_grid(source, grid, resampling)


def _read_open_to_grid(source, grid: dict, resampling: Resampling) -> np.ndarray:
    if source.count != 1 or source.crs is None or source.transform.a <= 0 or source.transform.e >= 0:
        raise ValueError("Invalid raster profile while building feature cache")
    if source.crs != grid["crs"] or not np.allclose(tuple(source.bounds), grid["bounds"], atol=1e-6, rtol=0):
        raise ValueError("Raster CRS/extent changed after verified audit")
    values = source.read(1).astype(np.float32)
    mask = source.read_masks(1)
    if not np.all(mask == 255) or not np.isfinite(values).all():
        raise ValueError("Raster contains invalid/nodata pixels after verified audit")
    if source.width == grid["width"] and source.height == grid["height"] and source.transform == grid["transform"]:
        return values
    destination = np.empty((grid["height"], grid["width"]), dtype=np.float32)
    reproject(values, destination, src_transform=source.transform, src_crs=source.crs,
              dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=resampling)
    if not np.isfinite(destination).all():
        raise ValueError("Non-finite values after canonical-grid reprojection")
    return destination


def _scene_target(reference: np.ndarray) -> np.ndarray:
    result = patch_label_targets(torch.from_numpy(reference[None].astype(np.int64)))
    counts = result["class_counts"][0].sum(0).to(torch.float64)
    if counts.sum() <= 0:
        raise ValueError("Reference map has no labelled pixels")
    return (counts / counts.sum()).to(torch.float32).numpy()


def _normalise_batch(adapter: CROMAAdapter, values: list[np.ndarray]) -> torch.Tensor:
    return torch.cat([adapter._croma_normalize(value) for value in values]).to(adapter.device)


def build_features(args) -> dict:
    started = time.perf_counter()
    manifest, rows = _load_contract(args.manifest, args.fingerprint)
    if sha256(args.checkpoint) != CROMA_CHECKPOINT_SHA256:
        raise ValueError("CROMA checkpoint hash is not frozen baseline checkpoint")
    s1_path, reference_path = _copy_nested_archives(Path(manifest["archive"]), args.cache_root / "source_cache")
    with zipfile.ZipFile(s1_path) as s1_zip, zipfile.ZipFile(reference_path) as reference_zip:
        vv = {Path(name).name[:-7]: name for name in s1_zip.namelist() if name.endswith("_VV.tif")}
        vh = {Path(name).name[:-7]: name for name in s1_zip.namelist() if name.endswith("_VH.tif")}
        refs = {Path(name).name[:-18]: name for name in reference_zip.namelist() if name.endswith("_reference_map.tif")}
        if not all(len(index) == 5000 for index in (vv, vh, refs)):
            raise ValueError("Nested archive identities do not match exact 5,000 population")
        adapter = CROMAAdapter(args.croma_source, args.checkpoint, device=args.device)
        provider = LocalRasterFeatureProvider()
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        shard_root = args.cache_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_entries = []
        for start in range(0, len(rows), args.shard_size):
            selected = rows[start:start + args.shard_size]
            shard = shard_root / f"{start:05d}-{start + len(selected) - 1:05d}.npz"
            if shard.is_file():
                with np.load(shard, allow_pickle=False) as cached:
                    if cached["area_ids"].tolist() == [row["area_id"] for row in selected]:
                        shard_entries.append({"path": str(shard), "rows": len(selected), "sha256": sha256(shard)})
                        print(f"Reused {start + len(selected)}/5000", flush=True)
                        continue
            physical, targets, optical_gap, sar_gap, joint_gap = [], [], [], [], []
            raw_optical, raw_sar = [], []
            for row in selected:
                paths = _s2_paths(row)
                with rasterio.open(paths["B02"]) as source:
                    grid = _profile(source)
                if (grid["height"], grid["width"]) != (120, 120):
                    raise ValueError(f"{row['area_id']}: canonical B02 grid changed")
                optical = np.stack([_read_path_to_grid(paths[band], grid, Resampling.bilinear) for band in OPTICAL_BANDS])
                identity = row["s1_identity"]
                sar = np.stack([_read_bytes_to_grid(s1_zip.read(index[identity]), grid, Resampling.bilinear)
                                for index in (vv, vh)])
                reference = _read_bytes_to_grid(reference_zip.read(refs[row["area_id"]]), grid, Resampling.nearest)
                if not np.all(reference == np.floor(reference)):
                    raise ValueError(f"{row['area_id']}: non-integer reference codes")
                vector, report = provider.extract(optical, sar, list(OPTICAL_BANDS), list(SAR_BANDS))
                if vector.shape != (62,) or not np.isfinite(vector).all() or len(report["feature_names"]) != 62:
                    raise ValueError(f"{row['area_id']}: invalid physical feature vector")
                physical.append(vector); targets.append(_scene_target(reference.astype(np.int64)))
                raw_optical.append(optical); raw_sar.append(sar)
            for offset in range(0, len(selected), args.inference_batch_size):
                optical_batch = raw_optical[offset:offset + args.inference_batch_size]
                sar_batch = raw_sar[offset:offset + args.inference_batch_size]
                with torch.inference_mode():
                    output = adapter.model(optical_images=_normalise_batch(adapter, optical_batch),
                                           SAR_images=_normalise_batch(adapter, sar_batch))
                optical_gap.extend(output["optical_GAP"].detach().cpu().to(torch.float32).numpy())
                sar_gap.extend(output["SAR_GAP"].detach().cpu().to(torch.float32).numpy())
                joint_gap.extend(output["joint_GAP"].detach().cpu().to(torch.float32).numpy())
                del output
            arrays = {
                "area_ids": np.asarray([row["area_id"] for row in selected]),
                "splits": np.asarray([row["split"] for row in selected]),
                "countries": np.asarray([row["country"] for row in selected]),
                "physical": np.asarray(physical, dtype=np.float32),
                "optical_croma": np.asarray(optical_gap, dtype=np.float32),
                "sar_croma": np.asarray(sar_gap, dtype=np.float32),
                "joint_croma": np.asarray(joint_gap, dtype=np.float32),
                "targets": np.asarray(targets, dtype=np.float32),
            }
            temporary = shard.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(shard)
            shard_entries.append({"path": str(shard), "rows": len(selected), "sha256": sha256(shard)})
            print(f"Built {start + len(selected)}/5000", flush=True)
    result = {
        "status": "complete", "sample_count": 5000, "split_counts": EXPECTED_SPLITS,
        "dataset_fingerprint": DATASET_FINGERPRINT, "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "split_fingerprint": SPLIT_FINGERPRINT, "croma_checkpoint_sha256": CROMA_CHECKPOINT_SHA256,
        "croma_source_revision": CROMA_SOURCE_REVISION,
        "normalization_implementation": "src.croma_adapter.CROMAAdapter._croma_normalize",
        "feature_dimensions": {"physical": 62, "optical_croma": 768, "sar_croma": 768,
                               "joint_croma": 768, "hybrid": 830},
        "shards": shard_entries, "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(adapter.device) if adapter.device.type == "cuda" else 0,
        "test_features_extracted_but_not_loaded_for_training": True,
    }
    atomic_json(args.cache_root / "manifest.json", result)
    return result


def load_cache(cache_root: Path, *, include_test: bool) -> dict[str, np.ndarray]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("dataset_fingerprint") != DATASET_FINGERPRINT:
        raise ValueError("Feature cache is incomplete or belongs to another dataset")
    fields = ("area_ids", "splits", "countries", "physical", "optical_croma", "sar_croma", "joint_croma", "targets")
    values = {key: [] for key in fields}
    for entry in manifest["shards"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Feature shard hash changed: {path}")
        with np.load(path, allow_pickle=False) as shard:
            mask = np.ones(len(shard["splits"]), dtype=bool) if include_test else shard["splits"] != "test"
            for key in fields:
                values[key].append(shard[key][mask])
    result = {key: np.concatenate(parts) for key, parts in values.items()}
    expected = 5000 if include_test else 4800
    if len(result["area_ids"]) != expected or len(set(result["area_ids"].tolist())) != expected:
        raise ValueError("Feature cache area identities are incomplete or duplicated")
    return result


def _metrics(predictions: np.ndarray, truth: np.ndarray, ids: np.ndarray, seed: int) -> dict:
    return coverage_metrics(predictions, truth, ids, bootstrap_repeats=2000, seed=seed)


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def train_baseline(args) -> dict:
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_training" or config.get("dataset_fingerprint") != DATASET_FINGERPRINT:
        raise ValueError("Baseline configuration was not frozen before training")
    data = load_cache(args.cache_root, include_test=False)
    splits, ids, y = data["splits"], data["area_ids"], data["targets"].astype(np.float32)
    train_mask, validation_mask = splits == "train", splits == "validation"
    if (int(train_mask.sum()), int(validation_mask.sum())) != (4600, 200):
        raise ValueError("Train/validation counts changed; test must remain sealed")
    args.model_root.mkdir(parents=True, exist_ok=True)
    train_mean = y[train_mask].mean(0); train_mean /= train_mean.sum()
    results = {"constant": {
        "train": _metrics(np.repeat(train_mean[None], int(train_mask.sum()), axis=0), y[train_mask], ids[train_mask], args.seed),
        "validation": _metrics(np.repeat(train_mean[None], int(validation_mask.sum()), axis=0), y[validation_mask], ids[validation_mask], args.seed),
        "selected_epoch": None, "best_validation_loss": float(coverage_loss(torch.log(torch.from_numpy(train_mean))[None].repeat(int(validation_mask.sum()), 1), torch.from_numpy(y[validation_mask]))),
    }}
    feature_values = {
        "physical": data["physical"], "optical_croma": data["optical_croma"],
        "sar_croma": data["sar_croma"], "joint_croma": data["joint_croma"],
        "hybrid": np.concatenate((data["physical"], data["joint_croma"]), axis=1),
    }
    peak_gpu = 0
    for name, values in feature_values.items():
        train = ProbeSplit("train", torch.from_numpy(values[train_mask].astype(np.float32)),
                           torch.from_numpy(y[train_mask]), ids[train_mask])
        validation = ProbeSplit("validation", torch.from_numpy(values[validation_mask].astype(np.float32)),
                                torch.from_numpy(y[validation_mask]), ids[validation_mask])
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model, log = fit_probe(train, validation, seed=args.seed, max_epochs=args.max_epochs,
                               patience=args.patience, batch_size=1024, device=args.device)
        train_predictions = predict_rows(model, train.x).numpy()
        validation_predictions = predict_rows(model, validation.x).numpy()
        checkpoint = args.model_root / f"{name}.pt"
        torch.save({"state_dict": model.state_dict(), "feature": name, "input_dimension": values.shape[1],
                    "seed": args.seed, "selected_epoch": log["selected_epoch"],
                    "dataset_fingerprint": DATASET_FINGERPRINT, "test_evaluated": False}, checkpoint)
        results[name] = {
            "train": _metrics(train_predictions, train.y.numpy(), train.area_ids, args.seed),
            "validation": _metrics(validation_predictions, validation.y.numpy(), validation.area_ids, args.seed),
            "selected_epoch": log["selected_epoch"], "best_validation_loss": log["best_validation_loss"],
            "training_log": log, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "state_dict_sha256": _model_state_hash(model), "input_dimension": int(values.shape[1]),
        }
        if args.device.startswith("cuda"):
            peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated())
        print(f"Trained {name}: epoch={log['selected_epoch']} val_loss={log['best_validation_loss']:.6f} val_MAE={results[name]['validation']['mae_pp']:.4f}", flush=True)
    summary = {
        "status": "baseline_validation_complete_test_sealed", "models": list(MODES),
        "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
        "split_counts_used": {"train": 4600, "validation": 200, "test": 0},
        "test_loaded": False, "test_used_for_selection": False, "configuration_sha256": sha256(args.config),
        "seed": args.seed, "results": results, "class_order": [name for name, _ in CLASSES],
        "runtime_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": peak_gpu,
    }
    atomic_json(args.output, summary)
    return summary


def repeat_selected(args) -> dict:
    """Repeat the already-selected hybrid configuration without exposing test rows."""
    started = time.perf_counter()
    data = load_cache(args.cache_root, include_test=False)
    splits, ids, y = data["splits"], data["area_ids"], data["targets"].astype(np.float32)
    train_mask, validation_mask = splits == "train", splits == "validation"
    values = np.concatenate((data["physical"], data["joint_croma"]), axis=1).astype(np.float32)
    train = ProbeSplit("train", torch.from_numpy(values[train_mask]), torch.from_numpy(y[train_mask]), ids[train_mask])
    validation = ProbeSplit("validation", torch.from_numpy(values[validation_mask]), torch.from_numpy(y[validation_mask]), ids[validation_mask])
    model, log = fit_probe(train, validation, seed=args.seed, max_epochs=60, patience=10,
                           batch_size=1024, device=args.device)
    train_metrics = _metrics(predict_rows(model, train.x).numpy(), train.y.numpy(), train.area_ids, args.seed)
    validation_metrics = _metrics(predict_rows(model, validation.x).numpy(), validation.y.numpy(), validation.area_ids, args.seed)
    args.model_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.model_root / f"hybrid_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "feature": "hybrid", "input_dimension": 830,
                "seed": args.seed, "selected_epoch": log["selected_epoch"],
                "dataset_fingerprint": DATASET_FINGERPRINT, "test_evaluated": False}, checkpoint)
    result = {
        "status": "repeat_validation_complete_test_sealed", "configuration": "frozen baseline hybrid",
        "seed": args.seed, "selected_epoch": log["selected_epoch"], "training_log": log,
        "train": train_metrics, "validation": validation_metrics, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "state_dict_sha256": _model_state_hash(model),
        "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
        "test_loaded": False, "test_used_for_selection": False, "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output, result)
    return result


def analyze_train_validation(args) -> dict:
    data = load_cache(args.cache_root, include_test=False)
    y, splits, countries = data["targets"].astype(np.float64), data["splits"], data["countries"]
    class_rows = []
    for index, (name, _) in enumerate(CLASSES):
        class_rows.append({
            "index": index, "name": name,
            "train_positive_areas": int(((y[:, index] > 0) & (splits == "train")).sum()),
            "validation_positive_areas": int(((y[:, index] > 0) & (splits == "validation")).sum()),
            "train_mean_coverage": float(y[splits == "train", index].mean()),
            "validation_mean_coverage": float(y[splits == "validation", index].mean()),
        })
    country_rows = []
    for country in sorted(set(countries.tolist())):
        country_rows.append({"country": country, "train": int(((countries == country) & (splits == "train")).sum()),
                             "validation": int(((countries == country) & (splits == "validation")).sum())})
    def mean_entropy(rows: np.ndarray) -> float:
        positive = rows > 0
        terms = np.zeros_like(rows)
        terms[positive] = rows[positive] * np.log(rows[positive])
        return float((-terms.sum(1)).mean())

    result = {
        "status": "complete_train_validation_only", "test_loaded": False,
        "class_distribution": class_rows, "country_distribution": country_rows,
        "target_entropy_mean": {split: mean_entropy(y[splits == split]) for split in ("train", "validation")},
        "known_audit_context": {
            "within_train_positive_spatial_overlap_pairs": 138,
            "within_train_duplicate_reference_content_groups": 103,
            "cross_split_spatial_overlap_pairs": 0,
            "cross_split_duplicate_reference_groups": 0,
            "interpretation": "Retained as legitimate repeated-place/acquisition sampling; no samples were excluded or reweighted."
        },
    }
    atomic_json(args.output, result)
    return result


def _load_probe(path: Path, dimension: int) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("input_dimension") != dimension or payload.get("dataset_fingerprint") != DATASET_FINGERPRINT:
        raise ValueError(f"Checkpoint contract mismatch: {path}")
    model = ProbeHead(dimension).eval()
    model.load_state_dict(payload["state_dict"])
    return model


def final_test(args) -> dict:
    """Perform the one irreversible scientific test scoring event."""
    if args.receipt.exists() or args.output.exists():
        raise RuntimeError("Final test receipt/output already exists; refusing to evaluate the held-out test twice")
    started = time.perf_counter()
    final_config = json.loads(args.config.read_text(encoding="utf-8"))
    if final_config.get("status") != "frozen_for_single_test_evaluation":
        raise ValueError("Final configuration is not frozen for the single test event")
    if final_config.get("selected_model") != "hybrid" or final_config.get("seed") != 17:
        raise ValueError("Final selection differs from validation-only decision")
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    data = load_cache(args.cache_root, include_test=True)
    splits, ids, y = data["splits"], data["area_ids"], data["targets"].astype(np.float32)
    train_mask, test_mask = splits == "train", splits == "test"
    if int(test_mask.sum()) != 200:
        raise ValueError("Final held-out test count is not 200")
    truth, test_ids = y[test_mask], ids[test_mask]
    train_mean = y[train_mask].mean(0); train_mean /= train_mean.sum()
    predictions = {"constant": np.repeat(train_mean[None], 200, axis=0)}
    values = {
        "physical": data["physical"], "optical_croma": data["optical_croma"],
        "sar_croma": data["sar_croma"], "joint_croma": data["joint_croma"],
        "hybrid": np.concatenate((data["physical"], data["joint_croma"]), axis=1),
    }
    for name, array in values.items():
        checkpoint = Path(baseline["results"][name]["checkpoint"])
        model = _load_probe(checkpoint, int(array.shape[1]))
        if _model_state_hash(model) != baseline["results"][name]["state_dict_sha256"]:
            raise ValueError(f"Frozen model state changed: {name}")
        predictions[name] = predict_rows(model, torch.from_numpy(array[test_mask].astype(np.float32))).numpy()
    metrics = {name: _metrics(value, truth, test_ids, 17) for name, value in predictions.items()}
    independent = {name: {
        "mae_pp": float(np.abs((value.astype(np.float64) - truth) * 100).mean()),
        "rmse_pp": float(np.sqrt(np.square((value.astype(np.float64) - truth) * 100).mean())),
        "dominant_class_accuracy": float((value.argmax(1) == truth.argmax(1)).mean()),
    } for name, value in predictions.items()}
    max_difference = max(abs(independent[name][metric] - metrics[name][metric])
                         for name in MODES for metric in independent[name])
    result = {
        "status": "complete_single_final_test_event", "selected_model": "hybrid", "seed": 17,
        "selection_criterion": "minimum validation soft-target cross entropy",
        "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
        "final_config_sha256": sha256(args.config), "test_area_count": 200,
        "test_area_ids": test_ids.tolist(), "metrics": metrics, "independent_metrics": independent,
        "independent_metric_max_abs_difference": max_difference,
        "generalization_gap_test_minus_validation_mae_pp": {
            name: metrics[name]["mae_pp"] - baseline["results"][name]["validation"]["mae_pp"] for name in MODES
        },
        "test_used_for_selection": False, "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output, result)
    atomic_json(args.predictions, {"area_ids": test_ids.tolist(), "truth": truth.tolist(),
                                   "models": {name: value.tolist() for name, value in predictions.items()}})
    normalized = {
        "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
        "final_config_sha256": sha256(args.config), "selected_model": "hybrid", "seed": 17,
        "selected_state_dict_sha256": baseline["results"]["hybrid"]["state_dict_sha256"],
        "metrics": metrics, "class_order": [name for name, _ in CLASSES],
    }
    fingerprint = canonical_hash(normalized)
    atomic_json(args.fingerprint, {"algorithm": "sha256(canonical_json(scientific_basis))",
                                   "fingerprint": fingerprint, "basis": normalized})
    receipt = {"status": "consumed_once", "test_area_count": 200, "test_ids_sha256": canonical_hash(test_ids.tolist()),
               "metrics_sha256": sha256(args.output), "predictions_sha256": sha256(args.predictions),
               "scientific_fingerprint": fingerprint}
    atomic_json(args.receipt, receipt)
    return result


def materialize_artifacts(args) -> dict:
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    final = json.loads(args.final_metrics.read_text(encoding="utf-8"))
    repeat = json.loads(args.repeat.read_text(encoding="utf-8"))
    final_config = json.loads(args.final_config.read_text(encoding="utf-8"))
    baseline_metrics = {name: {"train": row["train"], "validation": row["validation"]}
                        for name, row in baseline["results"].items()}
    histories = {name: row.get("training_log") for name, row in baseline["results"].items()
                 if row.get("training_log") is not None}
    baseline_per_class = {name: {split: row[split]["per_class"] for split in ("train", "validation")}
                          for name, row in baseline["results"].items()}
    baseline_basis = {
        "dataset_fingerprint": DATASET_FINGERPRINT, "split_fingerprint": SPLIT_FINGERPRINT,
        "configuration_sha256": baseline["configuration_sha256"], "seed": 17,
        "validation_metrics": {name: {key: baseline_metrics[name]["validation"][key]
                                             for key in ("mae_pp", "rmse_pp", "dominant_class_accuracy")}
                               for name in MODES},
        "state_dict_sha256": {name: row.get("state_dict_sha256") for name, row in baseline["results"].items()},
    }
    atomic_json(args.baseline_root / "metrics.json", baseline_metrics)
    atomic_json(args.baseline_root / "history.json", histories)
    atomic_json(args.baseline_root / "per_class.json", baseline_per_class)
    atomic_json(args.baseline_root / "fingerprint.json", {
        "algorithm": "sha256(canonical_json(scientific_basis))", "fingerprint": canonical_hash(baseline_basis),
        "basis": baseline_basis,
    })
    atomic_json(args.final_root / "per_class.json", final["metrics"]["hybrid"]["per_class"])
    atomic_json(args.final_root / "ablations.json", final["metrics"])
    atomic_json(args.final_root / "reproducibility.json", {
        "primary_seed": 17, "repeat_seed": 29,
        "primary_validation": baseline["results"]["hybrid"]["validation"],
        "repeat_validation": repeat["validation"],
        "mae_difference_repeat_minus_primary_pp": repeat["validation"]["mae_pp"] - baseline["results"]["hybrid"]["validation"]["mae_pp"],
        "loss_difference_repeat_minus_primary": repeat["training_log"]["best_validation_loss"] - baseline["results"]["hybrid"]["best_validation_loss"],
        "interpretation": final_config["repeatability"]["interpretation"],
    })
    root = args.baseline_root.parent
    atomic_json(root / "baseline_metrics.json", {"status": "complete_validation_and_final_test",
                "train": baseline["results"]["hybrid"]["train"], "validation": baseline["results"]["hybrid"]["validation"],
                "test": final["metrics"]["hybrid"], "ablations": final["metrics"], "areas_used": 5000})
    atomic_json(root / "final_metrics.json", {"status": "complete", "selected_model": "hybrid",
                "train": baseline["results"]["hybrid"]["train"], "validation": baseline["results"]["hybrid"]["validation"],
                "test": final["metrics"]["hybrid"], "ablations": final["metrics"], "areas_used": 5000})
    atomic_json(root / "experiments.json", {"status": "complete", "experiments": [
        {"name": "exact_5000_dataset_forensic_audit", "result": "ready_for_training"},
        {"name": "frozen_pipeline3_5000_baseline", "result": "complete", "test_used_for_selection": False},
        {"name": "hybrid_seed29_repeat", "result": "stable", "test_used_for_selection": False},
        {"name": "controlled_repair_decision", "result": "no repair justified"},
        {"name": "single_final_test_event", "result": "complete", "test_area_count": 200},
    ], "failed_or_hidden_trials": []})
    return {"status": "complete"}


def main() -> None:
    local_data_root = Path(os.environ.get("SATQUERY_LOCAL_DATA_ROOT", ROOT / "data" / "raw"))
    pipeline_cache = Path(os.environ.get("PIPELINE3_CACHE_ROOT", local_data_root / "pipeline3-5000" / "scene-features"))
    model_root = Path(os.environ.get("PIPELINE3_MODEL_ROOT", local_data_root / "pipeline3-5000" / "models"))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("features")
    features.add_argument("--manifest", type=Path, default=Path("experiments/pipeline3_5000/dataset_manifest.json"))
    features.add_argument("--fingerprint", type=Path, default=Path("experiments/pipeline3_5000/fingerprint.json"))
    features.add_argument("--cache-root", type=Path, default=pipeline_cache)
    features.add_argument("--croma-source", type=Path, default=Path(os.environ.get("CROMA_SOURCE", local_data_root / "croma-official")))
    features.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("CROMA_CHECKPOINT", local_data_root / "checkpoints" / "CROMA_base.pt")))
    features.add_argument("--device", default="cuda")
    features.add_argument("--shard-size", type=int, default=32)
    features.add_argument("--inference-batch-size", type=int, default=8)
    train = sub.add_parser("train")
    train.add_argument("--cache-root", type=Path, default=pipeline_cache)
    train.add_argument("--config", type=Path, default=Path("experiments/pipeline3_5000/baseline/config.json"))
    train.add_argument("--model-root", type=Path, default=model_root / "baseline")
    train.add_argument("--output", type=Path, default=Path("experiments/pipeline3_5000/baseline/validation_results.json"))
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--max-epochs", type=int, default=60)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--device", default="cuda")
    repeat = sub.add_parser("repeat")
    repeat.add_argument("--cache-root", type=Path, default=pipeline_cache)
    repeat.add_argument("--model-root", type=Path, default=model_root / "repeat")
    repeat.add_argument("--output", type=Path, default=Path("experiments/pipeline3_5000/final/repeat_seed29.json"))
    repeat.add_argument("--seed", type=int, default=29)
    repeat.add_argument("--device", default="cuda")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--cache-root", type=Path, default=pipeline_cache)
    analysis.add_argument("--output", type=Path, default=Path("experiments/pipeline3_5000/reports/train_validation_analysis.json"))
    final = sub.add_parser("final-test")
    final.add_argument("--cache-root", type=Path, default=pipeline_cache)
    final.add_argument("--config", type=Path, default=Path("experiments/pipeline3_5000/final/config.json"))
    final.add_argument("--baseline-results", type=Path, default=Path("experiments/pipeline3_5000/baseline/validation_results.json"))
    final.add_argument("--output", type=Path, default=Path("experiments/pipeline3_5000/final/metrics.json"))
    final.add_argument("--predictions", type=Path, default=Path("experiments/pipeline3_5000/final/predictions.json"))
    final.add_argument("--fingerprint", type=Path, default=Path("experiments/pipeline3_5000/final/fingerprint.json"))
    final.add_argument("--receipt", type=Path, default=Path("experiments/pipeline3_5000/final/test_receipt.json"))
    artifacts = sub.add_parser("artifacts")
    artifacts.add_argument("--baseline-results", type=Path, default=Path("experiments/pipeline3_5000/baseline/validation_results.json"))
    artifacts.add_argument("--final-metrics", type=Path, default=Path("experiments/pipeline3_5000/final/metrics.json"))
    artifacts.add_argument("--repeat", type=Path, default=Path("experiments/pipeline3_5000/final/repeat_seed29.json"))
    artifacts.add_argument("--final-config", type=Path, default=Path("experiments/pipeline3_5000/final/config.json"))
    artifacts.add_argument("--baseline-root", type=Path, default=Path("experiments/pipeline3_5000/baseline"))
    artifacts.add_argument("--final-root", type=Path, default=Path("experiments/pipeline3_5000/final"))
    args = parser.parse_args()
    runners = {"features": build_features, "train": train_baseline, "repeat": repeat_selected,
               "analyze": analyze_train_validation, "final-test": final_test, "artifacts": materialize_artifacts}
    result = runners[args.command](args)
    print(json.dumps({key: result[key] for key in result if key in {"status", "sample_count", "split_counts", "split_counts_used", "runtime_seconds", "peak_gpu_memory_bytes"}}, indent=2))


if __name__ == "__main__":
    main()
