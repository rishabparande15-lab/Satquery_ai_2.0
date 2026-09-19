"""Small deterministic acquisition boundary for selected remote artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


SCHEMA_VERSION = "acquisition_v1"
RangeReader = Callable[[str, int, int], bytes]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("relative_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("relative_path must stay inside the acquisition destination")
    return path.as_posix()


@dataclass(frozen=True)
class AcquisitionItem:
    item_id: str
    url: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("item_id is required")
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("url is required")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256.lower()):
            raise ValueError("sha256 must be a hexadecimal SHA-256 digest")
        object.__setattr__(self, "relative_path", _validate_relative_path(self.relative_path))


@dataclass(frozen=True)
class AcquisitionPlan:
    dataset: str
    dataset_version: str
    items: tuple[AcquisitionItem, ...]
    selection: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.dataset or not self.dataset_version:
            raise ValueError("dataset and dataset_version are required")
        items = tuple(sorted(self.items, key=lambda item: item.item_id))
        if len(items) != len({item.item_id for item in items}):
            raise ValueError("duplicate acquisition item_id")
        paths = [item.relative_path for item in items]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate acquisition relative_path")
        if not all(isinstance(item, AcquisitionItem) for item in items):
            raise ValueError("items must be AcquisitionItem values")
        object.__setattr__(self, "items", items)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "dataset": self.dataset,
                "dataset_version": self.dataset_version,
                "selection": dict(self.selection),
                "items": [asdict(item) for item in self.items]}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()


class AcquisitionError(RuntimeError):
    """Raised after a run records all item-level acquisition failures."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(f"acquisition failed for {len(report.get('failures', []))} item(s)")


def http_range_reader(url: str, start: int, length: int) -> bytes:
    """Read exactly one HTTP byte range; callers provide retry policy."""
    request = Request(url, headers={"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=45) as response:
        expected = f"bytes {start}-{start + length - 1}/*"
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(expected[:-1]):
            raise ValueError("source did not return the requested byte range")
        value = response.read(length + 1)
    if len(value) != length:
        raise OSError("incomplete range response")
    return value


def acquire_plan(
    plan: AcquisitionPlan,
    destination: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    attempts: int = 3,
    read_range: RangeReader = http_range_reader,
) -> dict[str, Any]:
    """Acquire only plan items with bounded memory and resumable partial files."""
    if type(chunk_size) is not int or chunk_size < 1 or type(attempts) is not int or attempts < 1:
        raise ValueError("chunk_size and attempts must be positive integers")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "plan.json", plan.to_dict() | {"plan_sha256": plan.fingerprint})
    receipts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in plan.items:
        target = destination / item.relative_path
        partial = target.with_name(target.name + ".part")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                if target.stat().st_size != item.size_bytes or _sha256(target) != item.sha256:
                    raise ValueError("completed file checksum or size mismatch")
                status = "reused"
            else:
                offset = partial.stat().st_size if partial.is_file() else 0
                if offset > item.size_bytes:
                    raise ValueError("partial file is larger than planned artifact")
                with partial.open("ab") as stream:
                    while offset < item.size_bytes:
                        length = min(chunk_size, item.size_bytes - offset)
                        value = None
                        for attempt in range(attempts):
                            try:
                                value = read_range(item.url, offset, length)
                                if len(value) != length:
                                    raise OSError("range reader returned an unexpected length")
                                break
                            except (OSError, ValueError):
                                if attempt == attempts - 1:
                                    raise
                        stream.write(value)
                        stream.flush()
                        os.fsync(stream.fileno())
                        offset += length
                if partial.stat().st_size != item.size_bytes or _sha256(partial) != item.sha256:
                    raise ValueError("download checksum or size mismatch")
                os.replace(partial, target)
                status = "downloaded"
            receipts.append({"item_id": item.item_id, "relative_path": item.relative_path,
                             "size_bytes": item.size_bytes, "sha256": item.sha256, "status": status})
        except Exception as error:
            failures.append({"item_id": item.item_id, "relative_path": item.relative_path,
                             "error": f"{type(error).__name__}: {error}"})
    report = {"schema_version": SCHEMA_VERSION, "status": "complete" if not failures else "failed",
              "dataset": plan.dataset, "dataset_version": plan.dataset_version,
              "plan_sha256": plan.fingerprint, "item_count": len(plan.items),
              "completed_count": len(receipts), "failures": failures,
              "receipts": receipts, "resumable": True, "chunk_size": chunk_size,
              "attempts": attempts}
    _atomic_json(destination / "acquisition_report.json", report)
    if failures:
        raise AcquisitionError(report)
    return report