"""Provider-neutral data discovery. Local data is explicitly labelled, never GEE data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import get_settings
from .dataset_loader import discover_samples


@dataclass(frozen=True)
class RetrievalResult:
    source: str
    status: str
    datasets: list[str]
    assets: dict[str, str]
    filters: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataOrchestrator:
    def retrieve(self, plan: dict[str, Any], request: dict[str, Any]) -> RetrievalResult:
        files = {key: str(value) for key, value in request.get("files", {}).items() if value}
        filters = {"aoi": plan.get("aoi"), "start_date": plan.get("start_date"), "end_date": plan.get("end_date"), "cloud_filter": request.get("cloud_cover")}
        if files:
            return RetrievalResult("user-provided upload", "available", plan.get("datasets", []), files, filters, [])
        sample_id = request.get("sample_id")
        settings = get_settings()
        if sample_id and settings.dataset_root.exists():
            try:
                # Full BigEarthNet collections must use metadata-backed complete
                # identities.  The suffix parser remains only for the legacy
                # three-sample fixture, which has no metadata table.
                strict = (settings.dataset_root / "metadata.parquet").is_file()
                ids = {sample.patch_id for sample in discover_samples(settings.dataset_root, strict=strict)}
            except (ValueError, OSError):
                return RetrievalResult("local development provider", "unavailable", plan.get("datasets", []), {}, filters, ["Local sample discovery failed; check the dataset configuration."])
            if sample_id in ids:
                return RetrievalResult("local development provider", "available", plan.get("datasets", []), {"sample_id": sample_id, "dataset_root": str(settings.dataset_root)}, filters, ["This is existing local development data; it was not retrieved from Google Earth Engine."])
        return RetrievalResult("external provider", "requires external data provider", plan.get("datasets", []), {}, filters, ["Configure GEE/authentication or provide uploaded imagery, or select a local development sample."])
