"""Sanitize machine-local paths at the public JSON boundary."""
from __future__ import annotations

import os
import re
from typing import Any, Iterable


_REDACTED = "[local path redacted]"
_WINDOWS_PATH = re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+)[^\s\"'<>]*")
_POSIX_MACHINE_PATH = re.compile(
    r"(?i)(?<![\w:/])/(?:home|tmp|private|users|mnt|media|var|opt|root|workspace|workspaces|app|srv|run|data|etc|proc|sys|dev)/[^\s\"'<>]*"
)
_HF_CACHE_PATH = re.compile(
    r"(?i)(?:huggingface[\\/](?:hub|cache)|models--[^\\/\s]+[\\/](?:snapshots|refs)[\\/][^\s\"'<>]+)"
)
_UPLOAD_STAGING_PATH = re.compile(
    r"(?i)(?:^|(?<=[\s=\"']))(?:experiments[\\/]+outputs[\\/]+web_uploads|web_uploads)[\\/]+(?:request|raster-inspection|raster-pair-inspection)-[a-z0-9_-]+(?:[\\/][^\s\"'<>]*)?"
)


def sanitize_public_value(value: Any, *, local_roots: Iterable[str | os.PathLike[str]] = ()) -> Any:
    """Recursively replace machine-local paths without stripping IDs like ``Qwen/model``."""
    if isinstance(value, dict):
        return {key: sanitize_public_value(item, local_roots=local_roots) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_public_value(item, local_roots=local_roots) for item in value]
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return value

    sanitized = _WINDOWS_PATH.sub(_REDACTED, value)
    sanitized = _POSIX_MACHINE_PATH.sub(_REDACTED, sanitized)
    sanitized = _HF_CACHE_PATH.sub(_REDACTED, sanitized)
    sanitized = _UPLOAD_STAGING_PATH.sub(_REDACTED, sanitized)
    for root in local_roots:
        root_text = os.fspath(root).rstrip("\\/")
        if not root_text:
            continue
        variants = {root_text, root_text.replace("\\", "/"), root_text.replace("/", "\\")}
        for variant in sorted(variants, key=len, reverse=True):
            sanitized = re.sub(re.escape(variant), _REDACTED, sanitized, flags=re.IGNORECASE)
    return sanitized