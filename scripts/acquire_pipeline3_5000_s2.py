"""Acquire only the missing optical records for the exact Pipeline 3 manifest.

The source and byte-range reader match the pinned acquisition route used for
the repository's earlier 1,000-area experiment.  Selection is never performed
here: keys come exclusively from the existing exact-5,000 audit manifest.
"""
from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import struct
import time
from urllib.request import Request, urlopen
import uuid

from affine import Affine
import numpy as np
import rasterio
from safetensors.numpy import load as load_safetensors


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "pipeline3_5000" / "dataset_manifest.json"
DEFAULT_DESTINATION = Path(
    os.environ.get("PIPELINE3_S2_DESTINATION", ROOT / "data" / "raw" / "pipeline3-5000" / "BigEarthNet-S2")
)
DEFAULT_WORK = Path(os.environ.get(
    "PIPELINE3_S2_WORK_ROOT",
    ROOT / "data" / "raw" / "pipeline3-5000-s2-acquisition",
))
LMDB_REVISION = "118d1b6285c080ba8e4078414e1b8a243b18c9bd"
LMDB_URL = (
    "https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB/resolve/"
    f"{LMDB_REVISION}/BENv2.lmdb/data.mdb"
)
LMDB_SIZE = 155_372_892_160
EXPECTED_BANDS = frozenset(
    {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}
)
EXPECTED_SHAPES = {
    "B01": (20, 20),
    "B02": (120, 120),
    "B03": (120, 120),
    "B04": (120, 120),
    "B05": (60, 60),
    "B06": (60, 60),
    "B07": (60, 60),
    "B08": (120, 120),
    "B8A": (60, 60),
    "B09": (20, 20),
    "B11": (60, 60),
    "B12": (60, 60),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class HTTPRanges:
    """Strict range client for the pinned, immutable LMDB object."""

    def __init__(self, url: str, size: int, *, attempts: int = 5):
        self.url, self.size, self.attempts = url, size, attempts

    def read(self, start: int, length: int) -> bytes:
        if start < 0 or length < 1 or start + length > self.size:
            raise ValueError("range outside pinned source")
        expected = f"bytes {start}-{start + length - 1}/{self.size}"
        for attempt in range(self.attempts):
            try:
                request = Request(
                    self.url,
                    headers={
                        "Range": f"bytes={start}-{start + length - 1}",
                        "User-Agent": "SatQuery-exact-5000/1.0",
                        "Accept-Encoding": "identity",
                    },
                )
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("source did not return the exact requested range")
                    value = response.read(length + 1)
                if len(value) != length:
                    raise OSError("incomplete range response")
                return value
            except (OSError, ValueError):
                if attempt == self.attempts - 1:
                    raise
                time.sleep(min(2**attempt, 16))
        raise AssertionError("unreachable")


class LMDBReader:
    """Read plain keys and inline/overflow values from the pinned LMDB layout.

    This is the deliberately narrow reader from raviasha/Sat_Query, retained
    here so this acquisition remains reproducible without a full LMDB download.
    """

    def __init__(self, read, size: int):
        self.read, self.size = read, size
        header = self.read(0, 4096)
        self.page_size = struct.unpack_from("<I", header, 40)[0]
        if self.page_size not in (4096, 16384):
            raise ValueError("unsupported LMDB page size")
        raw = self.read(0, 2 * self.page_size)
        versions = []
        for offset in (0, self.page_size):
            page = raw[offset : offset + self.page_size]
            if (
                struct.unpack_from("<II", page, 16) != (0xBEEFC0DE, 1)
                or struct.unpack_from("<H", page, 10)[0] != 8
                or struct.unpack_from("<I", page, 40)[0] != self.page_size
                or struct.unpack_from("<H", page, 92)[0] != 0
            ):
                raise ValueError("unsupported LMDB metadata layout")
            versions.append(
                (struct.unpack_from("<Q", page, 144)[0], struct.unpack_from("<Q", page, 128)[0])
            )
        self.transaction, self.root = max(versions)
        self.page = lru_cache(maxsize=4096)(self._page)

    def _page(self, number: int) -> bytes:
        if number < 2 or (number + 1) * self.page_size > self.size:
            raise ValueError("invalid LMDB page number")
        value = self.read(number * self.page_size, self.page_size)
        if len(value) != self.page_size or struct.unpack_from("<Q", value)[0] != number:
            raise ValueError("invalid LMDB page header")
        return value

    def get(self, key: str) -> bytes:
        encoded = key.encode()
        number = self.root
        for _ in range(20):
            page = self.page(number)
            flags, lower, upper = struct.unpack_from("<HHH", page, 10)
            if flags not in (1, 2) or not (16 <= lower <= upper <= self.page_size) or lower % 2:
                raise ValueError("unsupported or malformed LMDB page")
            nodes = []
            for index in range((lower - 16) // 2):
                offset = struct.unpack_from("<H", page, 16 + 2 * index)[0]
                if not upper <= offset <= self.page_size - 8:
                    raise ValueError("invalid LMDB node offset")
                low, high, node_flags, key_size = struct.unpack_from("<HHHH", page, offset)
                end = offset + 8 + key_size
                if end > self.page_size:
                    raise ValueError("invalid LMDB key length")
                nodes.append((page[offset + 8 : end], low, high, node_flags, end))
            keys = [node[0] for node in nodes]
            if keys != sorted(keys):
                raise ValueError("unordered LMDB keys")
            index = bisect.bisect_right(keys, encoded) - 1
            if index < 0:
                raise KeyError(key)
            found, low, high, node_flags, end = nodes[index]
            if flags == 1:
                number = low + (high << 16) + (node_flags << 32)
                continue
            if found != encoded:
                raise KeyError(key)
            length = low + (high << 16)
            if length > 2 * 1024 * 1024:
                raise ValueError("record exceeds supported size")
            if node_flags == 0:
                if end + length > self.page_size:
                    raise ValueError("invalid inline record")
                return page[end : end + length]
            if node_flags != 1 or end + 8 > self.page_size:
                raise ValueError("unsupported LMDB node flags")
            overflow = struct.unpack_from("<Q", page, end)[0]
            overflow_header = self.page(overflow)
            if struct.unpack_from("<H", overflow_header, 10)[0] != 4:
                raise ValueError("expected overflow page")
            pages = struct.unpack_from("<I", overflow_header, 12)[0]
            if length > pages * self.page_size - 16 or (overflow + pages) * self.page_size > self.size:
                raise ValueError("invalid overflow extent")
            return self.read(overflow * self.page_size + 16, length)
        raise ValueError("LMDB tree depth exceeded")


def missing_rows() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if len(rows) != 5000 or len({row["area_id"] for row in rows}) != 5000:
        raise ValueError("authoritative manifest is not the exact unique 5,000-area population")
    result = [row for row in rows if not row["validation"]["s2_valid"]]
    if not result:
        before = ROOT / "experiments" / "pipeline3_5000" / "s2_acquisition" / "missing_before.json"
        if before.is_file():
            before_ids = json.loads(before.read_text(encoding="utf-8"))["area_ids"]
            by_id = {row["area_id"]: row for row in rows}
            result = [by_id[area_id] for area_id in before_ids]
    if any(row["split"] != "train" for row in result):
        raise ValueError("missing S2 population no longer matches the established train-only gap")
    return result


def validate_completed(folder: Path, area_id: str) -> dict | None:
    receipt_path = folder / ".satquery-acquisition.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("area_id") != area_id or receipt.get("source", {}).get("revision") != LMDB_REVISION:
        return None
    files = receipt.get("bands", {})
    if set(files) != EXPECTED_BANDS:
        return None
    for band, item in files.items():
        path = folder / item["name"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            return None
    return receipt


def acquire_one(db: LMDBReader, destination: Path, work: Path, row: dict) -> dict:
    area_id = row["area_id"]
    if re.fullmatch(r"[A-Za-z0-9_]+", area_id) is None:
        raise ValueError(f"unsafe area ID: {area_id}")
    tile = area_id.rsplit("_", 2)[0]
    final = destination / tile / area_id
    completed = validate_completed(final, area_id)
    if completed:
        return {"status": "already_complete", "receipt": completed}
    if final.exists():
        raise ValueError(f"refusing to overwrite unverified destination: {final}")

    record_bytes = db.get(area_id)
    arrays = load_safetensors(record_bytes)
    if set(arrays) != EXPECTED_BANDS:
        raise ValueError(f"unexpected band set for {area_id}: {sorted(arrays)}")
    for band, array in arrays.items():
        if array.shape != EXPECTED_SHAPES[band] or array.dtype != np.dtype("uint16"):
            raise ValueError(f"invalid native raster for {area_id}/{band}: {array.shape} {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite raster values for {area_id}/{band}")

    temporary = work / f"{area_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        spatial = row["spatial"]
        base_transform = Affine(*spatial["transform"][:6])
        band_receipts = {}
        for band in sorted(EXPECTED_BANDS):
            array = arrays[band]
            height, width = array.shape
            path = temporary / f"{area_id}_{band}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=1,
                dtype="uint16",
                crs=spatial["crs"],
                transform=base_transform * Affine.scale(120 / width, 120 / height),
                nodata=0,
            ) as target:
                target.write(array, 1)
            with rasterio.open(path) as source:
                values = source.read(1)
                if (
                    source.count != 1
                    or source.shape != EXPECTED_SHAPES[band]
                    or source.dtypes != ("uint16",)
                    or source.crs is None
                    or source.crs.to_string() != spatial["crs"]
                    or [float(value) for value in source.bounds] != spatial["bounds"]
                    or not np.array_equal(values, array)
                ):
                    raise ValueError(f"written GeoTIFF failed round-trip validation: {area_id}/{band}")
                band_receipts[band] = {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "dimensions": [source.width, source.height],
                    "dtype": source.dtypes[0],
                    "crs": source.crs.to_string(),
                    "bounds": [float(value) for value in source.bounds],
                    "resolution": [abs(float(source.transform.a)), abs(float(source.transform.e))],
                    "nodata": source.nodata,
                    "finite": bool(np.isfinite(values).all()),
                    "north_up": bool(source.transform.a > 0 and source.transform.e < 0),
                }
        receipt = {
            "format_version": 1,
            "area_id": area_id,
            "s2_identity": row["s2_identity"],
            "split": row["split"],
            "source": {
                "dataset": "hackelle/BigEarthNetV2-LMDB",
                "status": "project-pinned unofficial pre-conversion mirror",
                "revision": LMDB_REVISION,
                "url": LMDB_URL,
                "record_key": area_id,
                "record_sha256": sha256_bytes(record_bytes),
            },
            "georeferencing_source": "exact selected manifest/reference-map spatial contract",
            "bands": band_receipts,
            "validation_status": "valid_native_arrays_and_geotiff_round_trip",
        }
        write_json_atomic(temporary / ".satquery-acquisition.json", receipt)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        return {"status": "acquired", "receipt": receipt}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    destination, work = args.destination.resolve(), args.work.resolve()
    if destination == work or destination in work.parents or work in destination.parents:
        raise ValueError("destination and work must be separate, non-nested directories")
    destination.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    rows = missing_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
    remote = HTTPRanges(LMDB_URL, LMDB_SIZE)
    db = LMDBReader(remote.read, LMDB_SIZE)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_one, db, destination, work, row): row["area_id"] for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result["receipt"])
            print(f"{result['status'].upper()} {index}/{len(rows)} {result['receipt']['area_id']}", flush=True)
    index = {
        "format_version": 1,
        "complete_for_requested_rows": len(results) == len(rows),
        "requested_count": len(rows),
        "source_revision": LMDB_REVISION,
        "area_ids": sorted(receipt["area_id"] for receipt in results),
    }
    write_json_atomic(destination.parent / "pipeline3-5000-s2-acquisition.json", index)
    print(f"COMPLETE {len(results)}/{len(rows)} exact manifest records", flush=True)


if __name__ == "__main__":
    main()
