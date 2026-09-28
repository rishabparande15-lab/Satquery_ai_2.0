"""Recover five authoritative S1 pairs for Phase 3H.1 structural validation.

This script never trains a model and never changes a scientific manifest.  It
reads only the already-pinned selected archive, writes ignored TIFF fixtures
atomically, and records a portable JSON provenance receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eo_vlm.sar_linkage import require_exact_s1_link, sar_annotation_applicability
from src.eo_vlm.sar_projector import SARProjector, normalize_sar

ARCHIVE_SHA256 = "b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595"
ARCHIVE_MEMBER = "bigearthnet-v2-5000/BigEarthNet-S1-selected.zip"
METADATA_MEMBER = "bigearthnet-v2-5000/metadata.parquet"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def raster_receipt(blob: bytes) -> tuple[np.ndarray, dict]:
    with MemoryFile(blob) as memory, memory.open() as source:
        values = source.read(1, masked=True)
        if source.count != 1 or source.width != 120 or source.height != 120:
            raise ValueError("invalid S1 raster dimensions/band count")
        if source.crs is None or source.transform.a <= 0 or source.transform.e >= 0:
            raise ValueError("invalid S1 CRS/orientation")
        if np.any(values.mask) or not np.isfinite(values.data).all():
            raise ValueError("invalid S1 nodata/non-finite values")
        return values.data.astype(np.float32), {
            "sha256": digest(blob), "dtype": source.dtypes[0], "nodata": source.nodata,
            "shape": [source.height, source.width], "crs": str(source.crs),
            "transform": list(source.transform), "bounds": list(source.bounds),
            "resolution": list(source.res), "valid_pixel_count": int(values.count()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("artifacts/annotations/bigearthnet_txt/annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/test_fixtures/s1_sar_vqa_fixture_manifest.json"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 3 or args.limit > 5:
        raise ValueError("fixture limit must be in [3,5]")
    if digest(args.archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("authoritative selected archive checksum mismatch")
    with zipfile.ZipFile(args.archive) as outer:
        metadata = pd.read_parquet(io.BytesIO(outer.read(METADATA_MEMBER)), columns=["patch_id", "s1_name", "split"])
        nested = outer.read(ARCHIVE_MEMBER)
    source = {row.patch_id: {"patch_id": row.patch_id, "s1_name": row.s1_name, "split": row.split}
              for row in metadata.itertuples(index=False)}
    candidates = []
    with args.annotations.open(encoding="utf-8") as stream:
        for line in stream:
            annotation = json.loads(line)
            if annotation.get("task_type") not in {"binary_qa", "multiple_choice_qa"} or annotation.get("split") == "test":
                continue
            item = source.get(annotation["image_id"])
            if item is None:
                continue
            require_exact_s1_link(annotation, item)
            if annotation.get("split") != item["split"]:
                raise ValueError("IMAGE_LINKAGE_BLOCKED: split mismatch")
            candidates.append((annotation, item))
            if len(candidates) == args.limit:
                break
    if len(candidates) != args.limit:
        raise RuntimeError("S1_ACQUISITION_BLOCKED: insufficient exact non-test linked records")
    fixture_dir = args.output.parent / "s1_raw"
    fixture_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="phase3h1-", dir=fixture_dir.parent))
    records = []
    try:
        with zipfile.ZipFile(io.BytesIO(nested)) as s1_zip:
            member_index = {Path(name).name: name for name in s1_zip.namelist() if name.endswith(".tif")}
            for annotation, item in candidates:
                s1 = item["s1_name"]
                names = {band: f"{s1}_{band}.tif" for band in ("VV", "VH")}
                if any(name not in member_index for name in names.values()):
                    raise RuntimeError("S1_ACQUISITION_BLOCKED: missing VV/VH source member")
                raw, profiles = {}, {}
                for band in ("VV", "VH"):
                    blob = s1_zip.read(member_index[names[band]])
                    raw[band], profiles[band] = raster_receipt(blob)
                    (staging / names[band]).write_bytes(blob)
                if profiles["VV"].keys() != profiles["VH"].keys() or any(profiles["VV"][key] != profiles["VH"][key] for key in ("shape", "crs", "transform", "bounds", "resolution")):
                    raise ValueError("VV/VH geospatial profiles differ")
                canonical = np.stack((raw["VV"], raw["VH"])); normalized = normalize_sar(canonical)
                with torch.no_grad():
                    output = SARProjector()(torch.from_numpy(normalized).unsqueeze(0))
                status, reason = sar_annotation_applicability(annotation)
                records.append({"area_id": item["patch_id"], "image_id": annotation["image_id"], "s1_identity": s1,
                    "annotation_id": annotation["annotation_id"], "split": item["split"], "channel_order": ["VV", "VH"],
                    "files": {band: {"filename": names[band], **profiles[band]} for band in ("VV", "VH")},
                    "canonical_tensor_sha256": digest(canonical.tobytes()), "canonical_shape": [2,120,120],
                    "normalization": "per-channel mean +/- 2 std; clipped [0,1]; nodata=0", "normalized_tensor_sha256": digest(normalized.tobytes()),
                    "projector_output_shape": list(output.shape), "annotation_applicability": status, "applicability_reason": reason})
        if fixture_dir.exists(): shutil.rmtree(fixture_dir)
        staging.replace(fixture_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = {"status": "SAR_VQA_SUPERVISION_BLOCKED", "source": {"identity": "project-pinned BigEarthNet v2 selected 5,000-area archive", "archive": str(args.archive), "archive_sha256": ARCHIVE_SHA256, "nested_s1_archive": ARCHIVE_MEMBER, "acquisition": "bounded local extraction from verified nested archive"}, "records": records, "annotation_audit": {"status": "SAR_UNKNOWN", "decision": "no SAR-only label applicability declaration in BigEarthNet.txt source record", "training_performed": False}, "scientific_representation_used": False}
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": receipt["status"], "samples": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
