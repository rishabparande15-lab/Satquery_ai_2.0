"""Fresh completeness audit for the exact selected 5,000-area collection.

This script is intentionally audit-only.  It never trains a model and stops the
scientific run when the selected population is not completely multimodal.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = Path(os.environ.get("SATQUERY_LOCAL_DATA_ROOT", ROOT / "data" / "raw"))
ARCHIVE = Path(os.environ.get(
    "PIPELINE3_5000_ARCHIVE",
    LOCAL_DATA_ROOT / "bigearthnet-v2-5000-20260911T162804Z-1-001.zip",
))
OUT = ROOT / "experiments" / "pipeline3_5000"
PREFIX = "bigearthnet-v2-5000/"
EXPECTED_ARCHIVE_SHA256 = "b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595"
EXPECTED_S2_BANDS = frozenset({"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"})
S2_ROOTS = (
    Path(os.environ.get("PIPELINE3_S2_EXISTING_ROOT", LOCAL_DATA_ROOT / "raw-1000" / "BigEarthNet-S2")),
    Path(os.environ.get("PIPELINE3_S2_DESTINATION", LOCAL_DATA_ROOT / "pipeline3-5000" / "BigEarthNet-S2")),
    Path(os.environ.get("DATASET_ROOT", LOCAL_DATA_ROOT / "bigearthnet-v2-small-sample")) / "BigEarthNet-S2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(name: str, value) -> str:
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def inspect_raster_bytes(content: bytes) -> dict:
    digest = hashlib.sha256(content).hexdigest()
    with MemoryFile(content) as memory:
        with memory.open() as source:
            array = source.read(1)
            mask = source.read_masks(1)
            transform = tuple(float(value) for value in source.transform)
            if source.nodata is None:
                mask_matches_nodata = True
            elif np.isnan(source.nodata):
                mask_matches_nodata = bool(np.array_equal(mask > 0, ~np.isnan(array)))
            else:
                mask_matches_nodata = bool(np.array_equal(mask > 0, array != source.nodata))
            result = {
                "sha256": digest,
                "width": source.width,
                "height": source.height,
                "count": source.count,
                "dtype": source.dtypes[0],
                "crs": source.crs.to_string() if source.crs else None,
                "transform": transform,
                "bounds": [float(value) for value in source.bounds],
                "resolution": [abs(float(source.transform.a)), abs(float(source.transform.e))],
                "finite": bool(np.isfinite(array).all()),
                "north_up": bool(source.transform.a > 0 and source.transform.e < 0 and source.transform.b == 0 and source.transform.d == 0),
                "nodata": source.nodata,
                "mask_flags": [flag.name for flags in source.mask_flag_enums for flag in flags],
                "valid_pixel_count": int(np.count_nonzero(mask)),
                "valid_pixel_fraction": float(np.count_nonzero(mask) / mask.size),
                "mask_matches_nodata": mask_matches_nodata,
            }
    return result


def inspect_raster_file(path: Path) -> dict:
    with path.open("rb") as stream:
        content = stream.read()
    return inspect_raster_bytes(content)


def aligned(*rasters: dict | None) -> bool:
    present = [item for item in rasters if item is not None]
    if not present:
        return False
    keys = ("width", "height", "crs", "transform", "bounds")
    return all(all(item[key] == present[0][key] for key in keys) for item in present[1:])


def discover_s2() -> tuple[dict[str, dict[str, Path]], dict[str, list[str]]]:
    by_area: dict[str, dict[str, Path]] = defaultdict(dict)
    sources: dict[str, list[str]] = defaultdict(list)
    for root in S2_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.tif"):
            stem = path.stem
            if "_B" not in stem:
                continue
            area_id, band = stem.rsplit("_", 1)
            if band not in EXPECTED_S2_BANDS:
                continue
            if band not in by_area[area_id]:
                by_area[area_id][band] = path
            sources[area_id].append(str(root))
    return dict(by_area), dict(sources)


def positive_spatial_overlaps(rows: list[dict]) -> list[dict]:
    by_crs: dict[str, list[tuple[list[float], str, str]]] = defaultdict(list)
    for row in rows:
        spatial = row.get("spatial") or {}
        if spatial.get("crs") and spatial.get("bounds"):
            by_crs[spatial["crs"]].append((spatial["bounds"], row["area_id"], row["split"]))
    overlaps = []
    for crs, items in by_crs.items():
        items.sort(key=lambda item: item[0][0])
        for index, (bounds, area_id, split) in enumerate(items):
            for other_bounds, other_id, other_split in items[index + 1:]:
                if other_bounds[0] >= bounds[2]:
                    break
                width = min(bounds[2], other_bounds[2]) - max(bounds[0], other_bounds[0])
                height = min(bounds[3], other_bounds[3]) - max(bounds[1], other_bounds[1])
                if width > 0 and height > 0:
                    overlaps.append({
                        "crs": crs,
                        "area_a": area_id,
                        "split_a": split,
                        "area_b": other_id,
                        "split_b": other_split,
                        "overlap_square_metres": width * height,
                        "cross_split": split != other_split,
                    })
    return overlaps


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"exact selected archive is unavailable: {ARCHIVE}")

    archive_hash_started = time.perf_counter()
    archive_sha256 = sha256_file(ARCHIVE)
    archive_hash_seconds = time.perf_counter() - archive_hash_started
    s2_by_area, s2_sources = discover_s2()

    with zipfile.ZipFile(ARCHIVE) as outer:
        outer_bad_member = outer.testzip()
        outer_names = outer.namelist()
        selection_bytes = outer.read(PREFIX + "selection.json")
        metadata_bytes = outer.read(PREFIX + "metadata.parquet")
        selection = json.loads(selection_bytes)
        metadata = pd.read_parquet(BytesIO(metadata_bytes))

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            nested_paths = {}
            for key, member in {
                "s1": PREFIX + "BigEarthNet-S1-selected.zip",
                "reference": PREFIX + "Reference_Maps-selected.zip",
            }.items():
                nested_path = temporary_root / Path(member).name
                with outer.open(member) as source, nested_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                nested_paths[key] = nested_path

            with zipfile.ZipFile(nested_paths["s1"]) as s1_zip, zipfile.ZipFile(nested_paths["reference"]) as reference_zip:
                s1_bad_member = s1_zip.testzip()
                reference_bad_member = reference_zip.testzip()
                s1_names = s1_zip.namelist()
                reference_names = reference_zip.namelist()
                vv_by_name = {Path(name).name[:-7]: name for name in s1_names if name.endswith("_VV.tif")}
                vh_by_name = {Path(name).name[:-7]: name for name in s1_names if name.endswith("_VH.tif")}
                reference_by_area = {
                    Path(name).name[:-18]: name for name in reference_names if name.endswith("_reference_map.tif")
                }

                rows = []
                s1_hashes: dict[tuple[str, str], str] = {}
                reference_hashes: dict[str, str] = {}
                s2_hashes: dict[tuple[str, str], str] = {}
                raster_started = time.perf_counter()
                for record in metadata.sort_values("patch_id").to_dict(orient="records"):
                    area_id = str(record["patch_id"])
                    s1_name = str(record["s1_name"])
                    vv_member = vv_by_name.get(s1_name)
                    vh_member = vh_by_name.get(s1_name)
                    reference_member = reference_by_area.get(area_id)
                    vv = inspect_raster_bytes(s1_zip.read(vv_member)) if vv_member else None
                    vh = inspect_raster_bytes(s1_zip.read(vh_member)) if vh_member else None
                    reference = inspect_raster_bytes(reference_zip.read(reference_member)) if reference_member else None
                    if vv:
                        s1_hashes[(area_id, "VV")] = vv["sha256"]
                    if vh:
                        s1_hashes[(area_id, "VH")] = vh["sha256"]
                    if reference:
                        reference_hashes[area_id] = reference["sha256"]

                    s2_files = s2_by_area.get(area_id, {})
                    s2_profiles = {}
                    for band, path in sorted(s2_files.items()):
                        profile = inspect_raster_file(path)
                        s2_profiles[band] = profile
                        s2_hashes[(area_id, band)] = profile["sha256"]

                    s1_present = vv is not None and vh is not None
                    s2_present = set(s2_files) == EXPECTED_S2_BANDS
                    reference_present = reference is not None
                    metadata_present = isinstance(record.get("labels"), (list, np.ndarray)) and len(record["labels"]) > 0
                    s1_valid = bool(s1_present and all(
                        item["width"] == 120 and item["height"] == 120 and item["count"] == 1
                        and item["finite"] and item["north_up"] and item["crs"]
                        for item in (vv, vh)
                    ) and aligned(vv, vh, reference))
                    reference_valid = bool(reference_present and reference["width"] == 120 and reference["height"] == 120
                                           and reference["count"] == 1 and reference["finite"] and reference["north_up"]
                                           and reference["crs"])
                    s2_reference_extent_aligned = bool(reference and s2_profiles and all(
                        item["crs"] == reference["crs"] and item["bounds"] == reference["bounds"]
                        for item in s2_profiles.values()
                    ))
                    s2_valid = bool(s2_present and all(
                        item["count"] == 1 and item["finite"] and item["north_up"] and item["crs"]
                        and item["valid_pixel_count"] > 0 and item["mask_matches_nodata"]
                        for item in s2_profiles.values()
                    ) and s2_reference_extent_aligned)
                    multimodal_valid = bool(s1_valid and s2_valid and reference_valid and metadata_present)
                    reasons = []
                    if not s1_present:
                        reasons.append("missing_s1_vv_or_vh")
                    elif not s1_valid:
                        reasons.append("invalid_or_misaligned_s1")
                    if not s2_present:
                        reasons.append("matching_s2_12_band_rasters_unavailable")
                    elif not s2_valid:
                        reasons.append("invalid_s2")
                    if not reference_present:
                        reasons.append("missing_reference_map")
                    elif not reference_valid:
                        reasons.append("invalid_reference_map")
                    if not metadata_present:
                        reasons.append("missing_or_empty_metadata_labels")
                    spatial = reference or vv or vh
                    rows.append({
                        "area_id": area_id,
                        "split": str(record["split"]),
                        "country": str(record["country"]),
                        "s1_identity": s1_name,
                        "s2_identity": str(record["s2v1_name"]),
                        "metadata_labels": list(record["labels"]),
                        "availability": {
                            "s1_vv": vv is not None,
                            "s1_vh": vh is not None,
                            "s2_expected_12_bands": s2_present,
                            "s2_band_count": len(s2_files),
                            "reference": reference_present,
                            "metadata": metadata_present,
                        },
                        "validation": {
                            "s1_valid": s1_valid,
                            "s2_valid": s2_valid,
                            "reference_valid": reference_valid,
                            "s1_reference_aligned": aligned(vv, vh, reference) if reference_present else False,
                            "s2_internal_alignment": aligned(*s2_profiles.values()) if s2_profiles else False,
                            "s2_reference_extent_aligned": s2_reference_extent_aligned,
                            "s2_native_multiresolution_expected": bool(s2_present and not aligned(*s2_profiles.values())),
                            "complete_multimodal": multimodal_valid,
                            "status": "valid" if multimodal_valid else "excluded_from_exact_multimodal_run",
                            "exclusion_reasons": reasons,
                        },
                        "spatial": None if spatial is None else {
                            "crs": spatial["crs"],
                            "dimensions": [spatial["height"], spatial["width"]],
                            "resolution": spatial["resolution"],
                            "transform": spatial["transform"],
                            "bounds": spatial["bounds"],
                            "orientation": "north_up" if spatial["north_up"] else "unsupported",
                        },
                        "content_sha256": {
                            "s1_vv": vv["sha256"] if vv else None,
                            "s1_vh": vh["sha256"] if vh else None,
                            "reference": reference["sha256"] if reference else None,
                            "s2_bands": {band: profile["sha256"] for band, profile in sorted(s2_profiles.items())},
                        },
                        "s2_native_band_metadata": {
                            band: {
                                "dimensions": [profile["height"], profile["width"]],
                                "resolution": profile["resolution"],
                                "crs": profile["crs"],
                                "finite": profile["finite"],
                                "north_up": profile["north_up"],
                                "dtype": profile["dtype"],
                                "nodata": profile["nodata"],
                                "mask_flags": profile["mask_flags"],
                                "valid_pixel_count": profile["valid_pixel_count"],
                                "valid_pixel_fraction": profile["valid_pixel_fraction"],
                                "mask_matches_nodata": profile["mask_matches_nodata"],
                            }
                            for band, profile in sorted(s2_profiles.items())
                        },
                        "s2_source_roots": sorted(set(s2_sources.get(area_id, []))),
                    })
                raster_seconds = time.perf_counter() - raster_started

    split_ids = {
        name: sorted(metadata.loc[metadata["split"] == name, "patch_id"].astype(str).tolist())
        for name in ("train", "validation", "test")
    }
    baseline_ids = {name: sorted(map(str, selection["baseline_ids"][name])) for name in split_ids}
    split_intersections = {
        "train_validation": sorted(set(split_ids["train"]) & set(split_ids["validation"])),
        "train_test": sorted(set(split_ids["train"]) & set(split_ids["test"])),
        "validation_test": sorted(set(split_ids["validation"]) & set(split_ids["test"])),
    }
    overlaps = positive_spatial_overlaps(rows)

    def duplicate_groups(values: dict) -> list[list[str]]:
        grouped = defaultdict(list)
        for identity, content_hash in values.items():
            grouped[content_hash].append(identity if isinstance(identity, str) else ":".join(identity))
        return [sorted(group) for group in grouped.values() if len(group) > 1]

    area_counts = Counter(row["area_id"] for row in rows)
    s1_identity_counts = Counter(row["s1_identity"] for row in rows)
    split_by_area = {row["area_id"]: row["split"] for row in rows}
    duplicate_s1_groups = duplicate_groups(s1_hashes)
    duplicate_reference_groups = duplicate_groups(reference_hashes)
    duplicate_s2_groups = duplicate_groups(s2_hashes)

    def cross_split_groups(groups: list[list[str]]) -> list[list[str]]:
        return [group for group in groups if len({split_by_area[item.split(":", 1)[0]] for item in group}) > 1]

    duplicate_reference_cross_split = cross_split_groups(duplicate_reference_groups)
    duplicate_s2_cross_split = cross_split_groups(duplicate_s2_groups)
    duplicate_s2_cross_area = [
        group for group in duplicate_s2_groups
        if len({item.split(":", 1)[0] for item in group}) > 1
    ]
    availability = {
        "total_selected_areas": len(rows),
        "complete_optical_areas": sum(row["validation"]["s2_valid"] for row in rows),
        "complete_sar_areas": sum(row["validation"]["s1_valid"] for row in rows),
        "complete_reference_areas": sum(row["validation"]["reference_valid"] for row in rows),
        "complete_metadata_areas": sum(row["availability"]["metadata"] for row in rows),
        "complete_multimodal_areas": sum(row["validation"]["complete_multimodal"] for row in rows),
        "complete_multimodal_by_split": {
            name: sum(row["validation"]["complete_multimodal"] and row["split"] == name for row in rows)
            for name in split_ids
        },
        "missing_optical_by_split": {
            name: sum(not row["validation"]["s2_valid"] and row["split"] == name for row in rows)
            for name in split_ids
        },
    }
    completeness_gate_passed = all(availability[key] == 5000 for key in (
        "total_selected_areas", "complete_optical_areas", "complete_sar_areas",
        "complete_reference_areas", "complete_metadata_areas", "complete_multimodal_areas",
    ))
    label_support = Counter(label for row in rows for label in row["metadata_labels"])
    leakage = {
        "area_id_duplicates": sorted(key for key, count in area_counts.items() if count > 1),
        "s1_identity_duplicates": sorted(key for key, count in s1_identity_counts.items() if count > 1),
        "split_intersections": split_intersections,
        "duplicate_s1_content_groups": duplicate_s1_groups,
        "duplicate_reference_content_groups": duplicate_reference_groups,
        "duplicate_reference_cross_split_groups": duplicate_reference_cross_split,
        "duplicate_s2_content_groups": duplicate_s2_groups,
        "duplicate_s2_cross_area_groups": duplicate_s2_cross_area,
        "duplicate_s2_cross_split_groups": duplicate_s2_cross_split,
        "positive_spatial_overlap_count": len(overlaps),
        "cross_split_positive_spatial_overlap_count": sum(item["cross_split"] for item in overlaps),
        "positive_spatial_overlaps": overlaps,
        "classification": {
            "intentional_design": [
                "The embedded baseline_ids identify the prior 600/200/200 multimodal subset; the full metadata split adds 4,000 exact selected training areas.",
                "Repeated geographic footprints and identical reference maps occur only within train and represent repeated-place/acquisition sampling, not cross-split leakage.",
            ],
            "unintentional_leakage_found": False,
            "caveat": (
                "Dataset-level checks passed; training-dependent leakage checks are deferred because Phase 6A.9 prohibits training."
                if completeness_gate_passed
                else "Training-dependent leakage checks were not reached because the completeness gate failed."
            ),
        },
        "target_derived_features": "not evaluated; no training or feature generation in this phase",
        "normalization_fit_scope": "not evaluated; no training or normalization fit in this phase",
        "persisted_representation_split_contamination": "not evaluated; no representations generated in this phase",
    }
    manifest_payload = {
        "format_version": 1,
        "status": "complete_for_availability_audit",
        "dataset": "BigEarthNet v2 exact selected 5,000-area subset",
        "archive": str(ARCHIVE),
        "archive_sha256": archive_sha256,
        "rows": rows,
    }
    dataset_manifest_hash = write_json("dataset_manifest.json", manifest_payload)
    split_payload = {
        "format_version": 1,
        "source": "embedded metadata.parquet and selection.json",
        "selection_seed": selection["seed"],
        "counts": {name: len(values) for name, values in split_ids.items()},
        "ids": split_ids,
        "declared_counts_match_metadata": selection["counts"] == {name: len(values) for name, values in split_ids.items()},
        "baseline_ids_are_subsets": all(set(baseline_ids[name]).issubset(split_ids[name]) for name in split_ids),
        "baseline_id_counts": {name: len(values) for name, values in baseline_ids.items()},
        "additional_id_counts": {name: len(set(split_ids[name]) - set(baseline_ids[name])) for name in split_ids},
        "intersections": split_intersections,
        "split_sha256": canonical_sha256(split_ids),
    }
    split_manifest_hash = write_json("split_manifest.json", split_payload)
    audit_summary = {
        "status": "blocked_before_training" if not completeness_gate_passed else "ready_for_training",
        "decision": "D" if not completeness_gate_passed else None,
        "reason": None if completeness_gate_passed else "The exact 5,000-area population has S1/reference/metadata for all areas but matching 12-band S2 for only 1,000; the additional 4,000 training areas are SAR-only.",
        "availability": availability,
        "archive": {
            "path": str(ARCHIVE),
            "size_bytes": ARCHIVE.stat().st_size,
            "sha256": archive_sha256,
            "expected_sha256_match": archive_sha256 == EXPECTED_ARCHIVE_SHA256,
            "outer_entries": len(outer_names),
            "outer_crc_bad_member": outer_bad_member,
            "s1_members": len(s1_names),
            "s1_crc_bad_member": s1_bad_member,
            "reference_members": len(reference_names),
            "reference_crc_bad_member": reference_bad_member,
            "s2_members": 0,
            "metadata_rows": len(metadata),
        },
        "split": {
            "counts": split_payload["counts"],
            "declared_counts_match_metadata": split_payload["declared_counts_match_metadata"],
            "baseline_ids_are_subsets": split_payload["baseline_ids_are_subsets"],
            "baseline_id_counts": split_payload["baseline_id_counts"],
            "additional_id_counts": split_payload["additional_id_counts"],
            "intersections_empty": all(not values for values in split_intersections.values()),
            "split_sha256": split_payload["split_sha256"],
        },
        "leakage": leakage,
        "class_label_area_support": dict(sorted(label_support.items())),
        "completeness_gate_passed": completeness_gate_passed,
        "training_started": False,
        "repair_attempted": completeness_gate_passed,
        "timing_seconds": {
            "archive_sha256": archive_hash_seconds,
            "raster_and_manifest_audit": raster_seconds,
            "total": time.perf_counter() - started,
        },
    }
    audit_summary_hash = write_json("audit_summary.json", audit_summary)

    not_run_metrics = {
        "status": "not_run",
        "reason": audit_summary["reason"] or "Training and test evaluation are outside Phase 6A.9 and did not run.",
        "areas_used": 0,
        "train": None,
        "validation": None,
        "test": None,
        "mae_pp": None,
        "rmse_pp": None,
        "dominant_class_accuracy": None,
        "per_class": None,
        "ablations": None,
    }
    write_json("baseline_metrics.json", not_run_metrics)
    write_json("final_metrics.json", not_run_metrics)
    write_json("experiments.json", {
        "status": "dataset_ready_training_not_started" if completeness_gate_passed else "stopped_at_dataset_completeness_gate",
        "experiments": [{
            "name": "exact_5000_dataset_forensic_audit",
            "result": audit_summary["status"],
            "training_started": False,
            "repair": "exact manifest-keyed S2 acquisition" if completeness_gate_passed else None,
        }],
        "failed_or_hidden_trials": [],
    })

    checkpoint = Path(os.environ.get("CROMA_CHECKPOINT", LOCAL_DATA_ROOT / "checkpoints" / "CROMA_base.pt"))
    reproducibility = {
        "status": audit_summary["status"],
        "code_commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "archive_sha256": archive_sha256,
        "metadata_parquet_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "selection_json_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "split_manifest_sha256": split_manifest_hash,
        "audit_summary_sha256": audit_summary_hash,
        "split_sha256": split_payload["split_sha256"],
        "selection_seed": selection["seed"],
        "preprocessing_version": "not_executed",
        "feature_schema": "not_executed",
        "training_configuration": None,
        "evaluation_configuration": None,
        "croma_checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "platform": platform.platform(),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reproducibility_hash = write_json("reproducibility.json", reproducibility)
    fingerprint_basis = {
        "status": audit_summary["status"],
        "archive_sha256": archive_sha256,
        "metadata_parquet_sha256": reproducibility["metadata_parquet_sha256"],
        "selection_json_sha256": reproducibility["selection_json_sha256"],
        "dataset_manifest_sha256": dataset_manifest_hash,
        "split_sha256": split_payload["split_sha256"],
        "availability": availability,
        "leakage_summary": {
            "area_id_duplicates": len(leakage["area_id_duplicates"]),
            "s1_identity_duplicates": len(leakage["s1_identity_duplicates"]),
            "duplicate_s1_content_groups": len(leakage["duplicate_s1_content_groups"]),
            "duplicate_reference_content_groups": len(leakage["duplicate_reference_content_groups"]),
            "duplicate_reference_cross_split_groups": len(leakage["duplicate_reference_cross_split_groups"]),
            "duplicate_s2_content_groups": len(leakage["duplicate_s2_content_groups"]),
            "duplicate_s2_cross_area_groups": len(leakage["duplicate_s2_cross_area_groups"]),
            "duplicate_s2_cross_split_groups": len(leakage["duplicate_s2_cross_split_groups"]),
            "cross_split_positive_spatial_overlap_count": leakage["cross_split_positive_spatial_overlap_count"],
        },
        "training_started": False,
    }
    write_json("fingerprint.json", {
        "status": "dataset_gate_fingerprint_only; no final model fingerprint exists",
        "algorithm": "sha256(canonical_json(fingerprint_basis))",
        "fingerprint": canonical_sha256(fingerprint_basis),
        "basis": fingerprint_basis,
        "reproducibility_sha256": reproducibility_hash,
    })
    print(json.dumps({
        "status": audit_summary["status"],
        "availability": availability,
        "split": audit_summary["split"],
        "archive": audit_summary["archive"],
        "leakage_counts": fingerprint_basis["leakage_summary"],
        "timing_seconds": audit_summary["timing_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
