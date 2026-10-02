"""Serialized, bounded GPU residency for frozen SatQuery specialists.

This owns controller lifetime only.  It never changes model construction
arguments, weights, prompts, preprocessing, or generation settings.
"""
from __future__ import annotations
from contextlib import contextmanager
import gc
import threading
from typing import Any, Callable

HEAVY_ROUTES = frozenset({"SINGLE_IMAGE_VQA", "SINGLE_IMAGE_SAR_VQA", "SINGLE_IMAGE_SCENE_DESCRIPTION", "OPTICAL_SAR_ANALYSIS", "TEMPORAL_CHANGE_DESCRIPTION"})

class SpecialistRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._controllers: dict[str, Any] = {}
        self._construction_counts: dict[str, int] = {}
        self._resident: str | None = None
        self._events: list[dict[str, str]] = []

    def _release_locked(self, route: str) -> None:
        controller = self._controllers.pop(route, None)
        if controller is None: return
        close = getattr(controller, "close", None)
        if callable(close): close()
        del controller
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
        if self._resident == route: self._resident = None
        self._events.append({"event": "released", "route": route})

    @contextmanager
    def use(self, route: str, factory: Callable[[], Any]):
        """Hold the sole heavy-runtime lock for a complete inference call."""
        with self._lock:
            if route in HEAVY_ROUTES and self._resident not in (None, route):
                self._release_locked(self._resident)
            if route not in self._controllers:
                self._controllers[route] = factory()
                self._construction_counts[route] = self._construction_counts.get(route, 0) + 1
                self._events.append({"event": "constructed", "route": route})
            if route in HEAVY_ROUTES: self._resident = route
            yield self._controllers[route]

    def release_all(self) -> None:
        with self._lock:
            for route in list(self._controllers): self._release_locked(route)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"resident_specialist": self._resident, "loaded_routes": sorted(self._controllers),
                    "construction_counts": dict(self._construction_counts), "recent_events": self._events[-12:]}

RUNTIME = SpecialistRuntime()
