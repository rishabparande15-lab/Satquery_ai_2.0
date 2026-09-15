"""Create exact-ID discovery and acquisition audit artifacts for Phase 6A.9."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = Path(os.environ.get("SATQUERY_LOCAL_DATA_ROOT", ROOT / "data" / "raw"))
PIPELINE = ROOT / "experiments" / "pipeline3_5000"
OUT = PIPELINE / "s2_acquisition"
MANIFEST = PIPELINE / "dataset_manifest.json"
EXPECTED_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_ROOTS = (
    Path(os.environ.get("PIPELINE3_S2_EXISTING_ROOT", LOCAL_DATA_ROOT / "raw-1000" / "BigEarthNet-S2")),
    Path(os.environ.get("PIPELINE3_S2_DESTINATION", LOCAL_DATA_ROOT / "pipeline3-5000" / "BigEarthNet-S2")),
    Path(os.environ.get("DATASET_ROOT", LOCAL_DATA_ROOT / "bigearthnet-v2-small-sample")) / "BigEarthNet-S2",
)
ARCHIVES = (
    Path(os.environ.get("BIGEARTHNET_SAMPLE_ARCHIVE", LOCAL_DATA_ROOT / "bigearthnet-v2-three-samples.zip")),
    Path(os.environ.get("PIPELINE3_5000_ARCHIVE", LOCAL_DATA_ROOT / "bigearthnet-v2-5000-20260911T162804Z-1-001.zip")),
    Path(os.environ.get("BIGEARTHNET_FULL_ARCHIVE", LOCAL_DATA_ROOT / "bigearthnet-v2-full-official-20260911T162841Z-1-031.zip")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value, *, compact: bool = False) -> None:
    options = {"sort_keys": True, "allow_nan": False}
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    path.write_text(json.dumps(value, **options) + "\n", encoding="utf-8")


def candidates() -> dict[str, dict[str, list[Path]]]:
    result: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for root in S2_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.tif"):
            if "_B" not in path.stem:
                continue
            area_id, band = path.stem.rsplit("_", 1)
            if band in EXPECTED_BANDS:
                result[area_id][band].append(path)
    return result


def archive_observations() -> list[dict]:
    observations = []
    for path in ARCHIVES:
        if not path.is_file():
            observations.append({"path": str(path), "status": "not_present"})
            continue
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            s2_tiffs = [name for name in names if name.lower().endswith((".tif", ".tiff")) and "BigEarthNet-S2" in name]
            opaque_parts = [name for name in names if "BigEarthNet-S2.tar.zst.parts/" in name]
            observations.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "zip_crc_bad_member": archive.testzip(),
                "member_count": len(names),
                "direct_s2_tiff_count": len(s2_tiffs),
                "opaque_s2_part_members": opaque_parts,
                "status": (
                    "candidate_s2_tiffs_present"
                    if s2_tiffs
                    else "rejected_no_independently_readable_s2_tiffs"
                ),
                "rejection_reason": (
                    None
                    if s2_tiffs
                    else (
                        "noncontiguous pieces of a split tar.zst cannot establish or extract exact patch identities"
                        if opaque_parts
                        else "archive contains no S2 TIFF payload"
                    )
                ),
            })
    return observations


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    selected = {row["area_id"]: row for row in rows}
    if len(rows) != 5000 or len(selected) != 5000:
        raise ValueError("dataset manifest is not the authoritative exact 5,000-area population")

    missing_path = OUT / "missing_before.json"
    if not missing_path.exists():
        missing = [
            {
                "area_id": row["area_id"],
                "s2_identity": row["s2_identity"],
                "split": row["split"],
                "required_bands": list(EXPECTED_BANDS),
                "expected_spatial": row["spatial"],
                "reason": "matching_s2_12_band_rasters_unavailable",
            }
            for row in rows
            if not row["validation"]["s2_valid"]
        ]
        if len(missing) != 4000:
            raise ValueError(f"expected the established 4,000-area pre-acquisition gap, found {len(missing)}")
        write_json(missing_path, {
            "format_version": 1,
            "source_manifest": str(MANIFEST.relative_to(ROOT)),
            "missing_count": len(missing),
            "area_ids": [item["area_id"] for item in missing],
            "records": missing,
        })

    found = candidates()
    matched = []
    unmatched = []
    duplicate_area_groups = []
    whole_area_hashes: dict[str, list[str]] = defaultdict(list)
    for area_id, bands in sorted(found.items()):
        all_paths = [path for paths in bands.values() for path in paths]
        duplicate_bands = {band: [str(path) for path in paths] for band, paths in bands.items() if len(paths) > 1}
        if duplicate_bands:
            duplicate_area_groups.append({"area_id": area_id, "duplicate_bands": duplicate_bands})
        canonical = {band: paths[0] for band, paths in bands.items()}
        if area_id not in selected:
            unmatched.append({
                "area_id": area_id,
                "paths": [str(path) for path in sorted(all_paths)],
                "band_count": len(canonical),
                "bands": sorted(canonical),
                "reason": "area_id_not_in_exact_selected_5000_manifest",
            })
            continue
        row = selected[area_id]
        band_hashes = {}
        band_sizes = {}
        dimensions = {}
        resolutions = {}
        nodata = {}
        valid_pixel_fractions = {}
        area_digest = hashlib.sha256()
        for band in EXPECTED_BANDS:
            path = canonical.get(band)
            if path is None:
                continue
            digest = sha256_file(path)
            area_digest.update(band.encode())
            area_digest.update(bytes.fromhex(digest))
            metadata = row.get("s2_native_band_metadata", {}).get(band, {})
            band_hashes[band] = digest
            band_sizes[band] = path.stat().st_size
            dimensions[band] = metadata.get("dimensions")
            resolutions[band] = metadata.get("resolution")
            nodata[band] = metadata.get("nodata")
            valid_pixel_fractions[band] = metadata.get("valid_pixel_fraction")
        complete = set(canonical) == set(EXPECTED_BANDS) and row["validation"]["s2_valid"]
        area_hash = area_digest.hexdigest() if len(band_hashes) == 12 else None
        acquisition_receipt_path = canonical[EXPECTED_BANDS[0]].parent / ".satquery-acquisition.json" if complete else None
        acquisition_receipt = (
            json.loads(acquisition_receipt_path.read_text(encoding="utf-8"))
            if acquisition_receipt_path and acquisition_receipt_path.is_file()
            else None
        )
        if area_hash:
            whole_area_hashes[area_hash].append(area_id)
        matched.append({
            "area_id": area_id,
            "s2_identity": row["s2_identity"],
            "split": row["split"],
            "logical_reference": str(canonical[EXPECTED_BANDS[0]].parent) if complete else None,
            "band_count": len(canonical),
            "band_sha256": band_hashes,
            "band_size_bytes": band_sizes,
            "dimensions_by_band": dimensions,
            "resolution_by_band": resolutions,
            "dtype": "uint16",
            "crs": row["spatial"]["crs"],
            "bounds": row["spatial"]["bounds"],
            "nodata_by_band": nodata,
            "valid_pixel_fraction_by_band": valid_pixel_fractions,
            "finite": all(item.get("finite") for item in row.get("s2_native_band_metadata", {}).values()),
            "north_up": all(item.get("north_up") for item in row.get("s2_native_band_metadata", {}).values()),
            "mask_matches_nodata": all(item.get("mask_matches_nodata") for item in row.get("s2_native_band_metadata", {}).values()),
            "area_content_sha256": area_hash,
            "acquisition_source_record_sha256": (
                acquisition_receipt["source"]["record_sha256"] if acquisition_receipt else None
            ),
            "acquisition_source_revision": (
                acquisition_receipt["source"]["revision"] if acquisition_receipt else None
            ),
            "reference_extent": row["spatial"],
            "validation_status": "valid" if complete else "rejected_incomplete_or_invalid",
        })

    content_duplicates = [
        {"area_content_sha256": digest, "area_ids": area_ids}
        for digest, area_ids in sorted(whole_area_hashes.items())
        if len(area_ids) > 1
    ]
    audit_path = PIPELINE / "audit_summary.json"
    leakage = json.loads(audit_path.read_text(encoding="utf-8"))["leakage"] if audit_path.is_file() else {}
    archives = archive_observations()
    valid_matches = [item for item in matched if item["validation_status"] == "valid"]
    write_json(OUT / "matched_s2.json", {
        "format_version": 1,
        "matched_count": len(valid_matches),
        "required_bands": list(EXPECTED_BANDS),
        "records": valid_matches,
    }, compact=True)
    write_json(OUT / "unmatched_candidates.json", {
        "format_version": 1,
        "unmatched_area_count": len(unmatched),
        "records": unmatched,
        "archive_observations": archives,
    })
    write_json(OUT / "duplicates.json", {
        "format_version": 1,
        "duplicate_area_band_groups": duplicate_area_groups,
        "duplicate_whole_area_content_groups": content_duplicates,
        "same_area_cross_band_content_groups": [
            group for group in leakage.get("duplicate_s2_content_groups", [])
            if len({item.split(":", 1)[0] for item in group}) == 1
        ],
        "cross_area_band_content_groups": leakage.get("duplicate_s2_cross_area_groups", []),
        "cross_split_duplicate_content_groups": [
            group for group in content_duplicates
            if len({selected[area_id]["split"] for area_id in group["area_ids"]}) > 1
        ] + leakage.get("duplicate_s2_cross_split_groups", []),
        "interpretation": "Same-area cross-band equality is recorded, not discarded. Cross-area and cross-split S2 duplication are the leakage checks.",
    })
    before_count = json.loads(missing_path.read_text(encoding="utf-8"))["missing_count"]
    write_json(OUT / "acquisition_status.json", {
        "format_version": 1,
        "selected_count": len(rows),
        "preexisting_valid_s2_count": len(rows) - before_count,
        "recovered_s2_count": before_count - (len(rows) - len(valid_matches)),
        "rejected_candidate_area_count": len(unmatched),
        "matched_valid_s2_count": len(valid_matches),
        "unresolved_selected_count": len(rows) - len(valid_matches),
        "candidate_roots": [str(path) for path in S2_ROOTS],
        "archives_inspected": archives,
        "split_counts": {
            split: sum(item["split"] == split for item in valid_matches)
            for split in ("train", "validation", "test")
        },
        "completeness_gate_ready": len(valid_matches) == 5000,
    })
    print(f"matched={len(valid_matches)} unmatched={len(unmatched)} unresolved={len(rows) - len(valid_matches)}")


if __name__ == "__main__":
    main()
