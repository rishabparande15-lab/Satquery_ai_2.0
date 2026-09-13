"""Pass 3 raw-area manifest and immutable split gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_loader import OPTICAL_BANDS, SAR_BANDS, discover_samples, load_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": str(path), "relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def build_manifest(dataset_root: Path, pipeline_root: Path | None = None) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    samples = discover_samples(dataset_root, strict=True)
    metadata = pd.read_parquet(dataset_root / "metadata.parquet")
    required_columns = {"patch_id", "s1_name", "split", "country"}
    if not required_columns.issubset(metadata.columns):
        raise ValueError(f"Metadata is missing columns: {sorted(required_columns - set(metadata.columns))}")
    by_patch = {str(row.patch_id): row for row in metadata.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        row = by_patch.get(sample.patch_id)
        status = "valid"
        error = None
        source_files = {"optical": {}, "sar": {}, "reference_map": None}
        try:
            if row is None:
                raise ValueError("area is absent from metadata")
            if str(row.s1_name) not in {path.parent.name for path in sample.sar_paths.values()}:
                raise ValueError("S1 identity does not match discovered source parent")
            if set(sample.optical_paths) != set(OPTICAL_BANDS) or set(sample.sar_paths) != set(SAR_BANDS) or not sample.reference_map.is_file():
                raise ValueError("incomplete optical, SAR, or reference-map sources")
            prepared = load_sample(sample)
            source_files = {
                "optical": {band: _source_record(sample.optical_paths[band], dataset_root) for band in OPTICAL_BANDS},
                "sar": {band: _source_record(sample.sar_paths[band], dataset_root) for band in SAR_BANDS},
                "reference_map": _source_record(sample.reference_map, dataset_root),
            }
            grid = {"shape": list(prepared.reference.shape), "crs": prepared.metadata["crs"],
                    "bounds": prepared.metadata["bounds"], "transform": prepared.metadata["transform"],
                    "resolution": prepared.metadata["resolution"]}
        except Exception as exc:
            status = "invalid"
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"area_id": sample.patch_id, "error": error})
            grid = None
        record = {"area_id": sample.patch_id, "s1_identity": str(getattr(row, "s1_name", "")) if row else None,
                  "s2_identity": sample.patch_id, "reference_identity": sample.reference_map.name,
                  "split": str(getattr(row, "split", "")) if row else None,
                  "country": str(getattr(row, "country", "")) if row else None, "status": status,
                  "source_files": source_files, "grid": grid}
        if error:
            record["error"] = error
        records.append(record)
    if failures:
        raise ValueError(f"Raw dataset validation failed for {len(failures)} areas: {failures[:3]}")
    counts = {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")}
    if counts != {"train": 600, "validation": 200, "test": 200}:
        raise ValueError(f"Expected canonical 600/200/200 split, found {counts}")
    split = {name: [record["area_id"] for record in records if record["split"] == name]
             for name in ("train", "validation", "test")}
    all_ids = [area_id for values in split.values() for area_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Area identity leakage across canonical splits")
    manifest = {"format_version": 1, "status": "validated", "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_root": str(dataset_root), "dataset": "BigEarthNet v2 pinned reference subset-1000",
                "area_count": len(records), "split_counts": counts, "samples": records,
                "split": {"seed": "official_metadata_split", "unit": "geographic_area", "areas": split},
                "preprocessing": {"optical_band_order": list(OPTICAL_BANDS), "sar_band_order": list(SAR_BANDS),
                                  "reference_grid": "B02", "normalization": "croma_readme_patch_8bit_v1",
                                  "resampling": "bilinear optical/SAR to B02; nearest reference"},
                "pipeline_artifacts": {}}
    if pipeline_root is not None:
        pipeline_root = Path(pipeline_root)
        for name in ("prepared", "features", "targets"):
            artifact_manifest = pipeline_root / name / "manifest.json"
            validation = pipeline_root / name / "validation.json"
            if not artifact_manifest.is_file() or not validation.is_file():
                raise FileNotFoundError(f"Missing pipeline artifact receipt: {artifact_manifest}")
            manifest["pipeline_artifacts"][name] = {"manifest": str(artifact_manifest), "manifest_sha256": _sha256(artifact_manifest),
                                                     "validation": json.loads(validation.read_text(encoding="utf-8"))}
    return manifest


def write_manifest(dataset_root: Path, output_root: Path, pipeline_root: Path | None = None) -> tuple[Path, Path]:
    manifest = build_manifest(dataset_root, pipeline_root)
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "dataset_manifest.json"
    split_path = output_root / "split.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps({"format_version": 1, "dataset_manifest": manifest_path.name, "seed": manifest["split"]["seed"],
                                      "unit": "geographic_area", "counts": manifest["split_counts"], "areas": manifest["split"]["areas"]},
                                     indent=2) + "\n", encoding="utf-8")
    return manifest_path, split_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw BigEarthNet areas and write the Pass 3 manifest/split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/pass3"))
    args = parser.parse_args()
    manifest, split = write_manifest(args.dataset_root, args.output_root, args.pipeline_root)
    print(json.dumps({"status": "validated", "manifest": str(manifest), "split": str(split)}, indent=2))


if __name__ == "__main__":
    main()