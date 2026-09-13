"""Shared GEE availability helpers for the preserved pipeline experiments.

This module does not retrieve imagery and is not the web API controller.  It
exists so the canonical GEE+CROMA experiment does not import implementation
details from the retired GEE-only experiment.
"""
from __future__ import annotations

import importlib.util
from typing import Any


def gee_status() -> dict[str, Any]:
    """Return the same explicit package/authentication state as the old runner."""
    available = importlib.util.find_spec("ee") is not None
    status: dict[str, Any] = {
        "package_available": available,
        "authenticated": False,
        "error": None,
    }
    if not available:
        status["error"] = "earthengine-api is not installed"
        return status
    try:
        import ee

        ee.Initialize()
        status["authenticated"] = True
    except Exception as error:
        status["error"] = f"GEE initialization failed: {error}"
    return status


def select_mode(requested: str) -> tuple[str, dict[str, Any]]:
    """Select GEE or the labelled local-reference mode without silent claims."""
    status = gee_status()
    if requested == "local":
        return "local_reference", status
    if requested == "gee":
        if not status["authenticated"]:
            raise RuntimeError(status["error"] or "GEE authentication unavailable")
        return "gee", status
    return ("gee", status) if status["authenticated"] else ("local_reference", status)
