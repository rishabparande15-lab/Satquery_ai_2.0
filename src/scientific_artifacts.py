"""Deterministic, provenance-aware storage for small scientific feature runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_artifact_bundle(root: Path, *, sample_id: str, arrays: Mapping[str, np.ndarray], provenance: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for name, value in arrays.items():
        if not name or Path(name).name != name or Path(name).suffix:
            raise ValueError(f"Invalid artifact array name: {name!r}")
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"Artifact array must be finite numeric data: {name}")
        path = root / f"{name}.npy"
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        manifest[name] = {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256(path)}
    receipt = {"artifact_version": 1, "status": "complete", "sample_id": sample_id,
               "arrays": manifest, "provenance": dict(provenance)}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def verify_artifact_bundle(root: Path) -> dict[str, Any]:
    root = Path(root)
    try:
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Artifact receipt is unreadable: {root}") from error
    if receipt.get("status") != "complete" or not receipt.get("sample_id"):
        raise ValueError("Artifact receipt is not complete")
    for name, metadata in receipt.get("arrays", {}).items():
        path = root / metadata["file"]
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Artifact hash mismatch: {name}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Artifact cannot be reloaded: {name}") from error
        if list(array.shape) != metadata.get("shape") or str(array.dtype) != metadata.get("dtype") or not np.isfinite(array).all():
            raise ValueError(f"Artifact contract mismatch: {name}")
    return {**receipt, "status": "verified"}