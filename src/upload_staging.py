"""Bounded classification and recovery for SatQuery-owned upload staging."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Iterable


ACTIVE = "ACTIVE"
RECENT = "RECENT"
STALE_SAFE_TO_DELETE = "STALE_SAFE_TO_DELETE"
UNKNOWN = "UNKNOWN_DO_NOT_DELETE"
_NAME = re.compile(r"^(?:request|raster-inspection|raster-pair-inspection)-[a-z0-9_]{8}$", re.IGNORECASE)
_IMAGE = re.compile(r"^(?:image|raster|first|second)\.(?:tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def _is_reparse(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
        attrs = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return path.is_symlink() or bool(attrs & reparse_flag) or stat.S_ISLNK(mode)
    except OSError:
        return True


def _active_set(active_paths: Iterable[str | os.PathLike[str]]) -> set[str]:
    return {os.path.normcase(os.path.abspath(os.fspath(path))) for path in active_paths}


def _expected_contents(path: Path) -> bool:
    prefix = path.name.rsplit("-", 1)[0].lower()
    names: list[str] = []
    try:
        for child in path.iterdir():
            if _is_reparse(child) or not child.is_file() or not _IMAGE.fullmatch(child.name):
                return False
            names.append(child.name.lower())
    except OSError:
        return False
    if prefix == "request":
        return len(names) <= 1 and (not names or names[0].startswith("image."))
    if prefix == "raster-inspection":
        return len(names) <= 1 and (not names or names[0].startswith("raster."))
    if prefix == "raster-pair-inspection":
        return len(names) <= 2 and len({name.split(".", 1)[0] for name in names}) == len(names) and all(
            name.startswith(("first.", "second.")) for name in names
        )
    return False


def classify_staging_directory(
    staging_root: str | os.PathLike[str],
    candidate: str | os.PathLike[str],
    *,
    ttl_seconds: float,
    active_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, object]:
    """Classify one immediate child; uncertainty always produces UNKNOWN."""
    root = Path(staging_root)
    target = Path(candidate)
    if not target.is_absolute():
        target = root / target
    result: dict[str, object] = {"name": target.name, "classification": UNKNOWN, "reason": "unverified"}
    try:
        if _is_reparse(root):
            result["reason"] = "staging root is a reparse point"
            return result
        resolved_root = root.resolve(strict=True)
        resolved_target = target.resolve(strict=True)
        if _is_reparse(resolved_root) or not resolved_root.is_dir():
            result["reason"] = "staging root is not a normal directory"
            return result
        if target.name in {"", ".", ".."} or target.parent.resolve(strict=True) != resolved_root:
            result["reason"] = "candidate is not an immediate child of staging root"
            return result
        if resolved_target.parent != resolved_root:
            result["reason"] = "resolved candidate escapes staging root"
            return result
        if _is_reparse(target) or not target.is_dir():
            result["reason"] = "candidate is not a normal directory"
            return result
        if not _NAME.fullmatch(target.name) or not _expected_contents(target):
            result["reason"] = "name or contents are not recognized SatQuery staging"
            return result
        canonical = os.path.normcase(os.path.abspath(str(resolved_target)))
        if canonical in _active_set(active_paths):
            result["classification"] = ACTIVE
            result["reason"] = "owned by an active request"
            return result
        age = time.time() - target.stat().st_mtime
        if age < ttl_seconds:
            result["classification"] = RECENT
            result["reason"] = "within configured staging TTL"
            return result
        result["classification"] = STALE_SAFE_TO_DELETE
        result["reason"] = "recognized, contained, inactive, expired staging"
        return result
    except (OSError, RuntimeError, ValueError):
        result["reason"] = "path could not be safely resolved or inspected"
        return result


def recover_staging(
    staging_root: str | os.PathLike[str],
    *,
    ttl_seconds: float,
    active_paths: Iterable[str | os.PathLike[str]] = (),
    delete_stale: bool = True,
) -> list[dict[str, object]]:
    """Inspect only immediate root children and remove only proven stale staging."""
    root = Path(staging_root)
    try:
        # Do this before resolve(): resolving first follows a configured root
        # link and could make an unrelated directory look like our staging root.
        if _is_reparse(root):
            return []
        resolved_root = root.resolve(strict=True)
        if _is_reparse(resolved_root) or not resolved_root.is_dir():
            return []
        children = list(resolved_root.iterdir())
    except (OSError, RuntimeError, ValueError):
        return []

    active_paths = tuple(active_paths)
    decisions = []
    for child in children:
        decision = classify_staging_directory(
            resolved_root, child, ttl_seconds=ttl_seconds, active_paths=active_paths
        )
        decision["deleted"] = False
        if delete_stale and decision["classification"] == STALE_SAFE_TO_DELETE:
            try:
                verified = classify_staging_directory(
                    resolved_root, child, ttl_seconds=ttl_seconds, active_paths=active_paths
                )
                if verified["classification"] != STALE_SAFE_TO_DELETE:
                    decision["classification"] = verified["classification"]
                    decision["reason"] = "classification changed before deletion"
                elif child.parent.resolve(strict=True) != resolved_root or child.resolve(strict=True).parent != resolved_root:
                    decision["classification"] = UNKNOWN
                    decision["reason"] = "containment changed before deletion"
                elif _is_reparse(child):
                    decision["classification"] = UNKNOWN
                    decision["reason"] = "candidate became a reparse point before deletion"
                else:
                    shutil.rmtree(child)
                    decision["deleted"] = True
            except OSError:
                decision["reason"] = "stale cleanup failed; directory preserved"
        decisions.append(decision)
    return decisions
