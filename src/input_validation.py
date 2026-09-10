"""Central validation for AOI, dates, and bounded GeoTIFF inputs."""
from datetime import date
import math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

TEMPORAL_MESSAGE = "Temporal analysis requires a valid spatially corresponding before-and-after image pair."
MAX_PIXELS = 4_000_000

def _check_date(value):
    return date.fromisoformat(value).isoformat() if value else None

def validate_aoi(aoi):
    if aoi is None:
        return ["AOI not supplied; the selected raster extent is used."]
    if not isinstance(aoi, dict):
        raise ValueError("AOI must be a point or bounding-box object.")
    def number(v):
        return type(v) in (float, int) and math.isfinite(v)
    if aoi.get("type") == "point":
        lat, lon = aoi.get("latitude"), aoi.get("longitude")
        if not number(lat) or not number(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("AOI point latitude/longitude is invalid.")
    elif aoi.get("type") == "bbox":
        b = aoi.get("bounds")
        if not isinstance(b, list) or len(b) != 4 or not all(number(v) for v in b) or not (-180 <= b[0] < b[2] <= 180 and -90 <= b[1] < b[3] <= 90):
            raise ValueError("AOI bbox must be [west, south, east, north] in valid longitude/latitude ranges.")
    else:
        raise ValueError("AOI type must be point or bbox; polygon processing is unavailable.")
    return []

def inspect_raster(path):
    if Path(path).suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF rasters are supported.")
    try:
        with rasterio.open(path) as ds:
            if ds.driver != "GTiff" or ds.width * ds.height * ds.count > MAX_PIXELS:
                raise ValueError("Raster must be a GeoTIFF with at most 4 million band-pixels.")
            values = ds.read(out_dtype="float32", masked=True).filled(np.nan)
            return {"readable": True, "format": ds.driver, "band_count": ds.count,
                    "crs": str(ds.crs) if ds.crs else None, "resolution": list(ds.res),
                    "shape": [ds.height, ds.width], "bounds": list(ds.bounds),
                    "transform": list(ds.transform), "finite": bool(np.isfinite(values).all()),
                    "band_descriptions": list(ds.descriptions), "acquisition_date": ds.tags().get("acquisition_date"),
                    "cloud_cover": ds.tags().get("cloud_cover"), "sar_units": ds.tags().get("sar_units", "unknown")}
    except rasterio.errors.RasterioError:
        raise ValueError("Raster is corrupt, missing, or unreadable.") from None

def check_grid(first, second):
    return {"crs": first["crs"] == second["crs"],
            "resolution": bool(np.allclose(first["resolution"], second["resolution"], rtol=0, atol=1e-8)),
            "extent": bool(np.allclose(first["bounds"], second["bounds"], rtol=0, atol=1e-6)),
            "transform": bool(np.allclose(first["transform"], second["transform"], rtol=0, atol=1e-8)),
            "shape": first["shape"] == second["shape"]}

def check_aoi_intersection(aoi, metadata):
    if not aoi:
        return
    b = transform_bounds(metadata["crs"], "EPSG:4326", *metadata["bounds"])
    if aoi["type"] == "point":
        intersects = b[0] <= aoi["longitude"] <= b[2] and b[1] <= aoi["latitude"] <= b[3]
    else:
        x = aoi["bounds"]
        intersects = x[0] < b[2] and x[2] > b[0] and x[1] < b[3] and x[3] > b[1]
    if not intersects:
        raise ValueError("Requested AOI does not intersect the selected raster.")

def validate_inputs(plan, retrieval, request):
    checks = {"aoi": True, "dates": True, "files": {}, "temporal_compatibility": None,
              "optical_sar_compatibility": None, "cloud_cover": "unknown; no cloud mask supplied"}
    errors, warnings = [], []
    try:
        warnings.extend(validate_aoi(plan.get("aoi")))
    except ValueError as exc:
        checks["aoi"] = False
        errors.append(str(exc))
    try:
        start, end = _check_date(plan.get("start_date")), _check_date(plan.get("end_date"))
        if start and end and start > end:
            raise ValueError("Start date must not be later than end date.")
    except (ValueError, TypeError):
        checks["dates"] = False
        errors.append("Dates must be valid ISO dates with start no later than end.")
    if retrieval.get("status") != "available":
        errors.append("No usable imagery is available. Select a local sample or upload the required rasters.")
    assets = retrieval.get("assets", {})
    if retrieval.get("source") == "user-provided upload":
        required = ["before", "after"] if plan.get("requires_temporal_pair") else plan["modalities"]
        for role in required:
            if role not in assets:
                errors.append(f"Missing required {role} raster.")
        for role, path in assets.items():
            try:
                meta = inspect_raster(path)
                checks["files"][role] = meta
                expected = 2 if role == "sar" else 12
                if meta["band_count"] != expected:
                    raise ValueError(f"{role} requires {expected} bands in canonical order.")
                if not meta["crs"] or not all(v > 0 for v in meta["resolution"]):
                    raise ValueError(f"{role} requires a CRS and positive resolution.")
                if not meta["finite"]:
                    raise ValueError(f"{role} has missing or non-finite pixels.")
                if checks["aoi"]:
                    check_aoi_intersection(plan.get("aoi"), meta)
            except ValueError as exc:
                errors.append(str(exc))
        for a, b, key in (("optical", "sar", "optical_sar_compatibility"), ("before", "after", "temporal_compatibility")):
            if a in checks["files"] and b in checks["files"]:
                compatibility = check_grid(checks["files"][a], checks["files"][b])
                checks[key] = compatibility
                if not all(compatibility.values()):
                    errors.append(f"{a}/{b} rasters have incompatible CRS, resolution, extent, transform or shape.")
    if plan.get("requires_temporal_pair"):
        paired = checks["files"].get("before"), checks["files"].get("after")
        valid_pair = False
        if all(paired):
            try:
                dates = [_check_date(m["acquisition_date"]) for m in paired]
                valid_pair = bool(all(dates) and dates[0] < dates[1] and all(check_grid(*paired).values()))
            except (ValueError, TypeError):
                pass
        checks["temporal_pair_valid"] = valid_pair
        if not valid_pair:
            errors.append(TEMPORAL_MESSAGE)
    return {"valid": not errors, "checks": checks, "errors": errors, "warnings": warnings,
            "compatibility": {"source": retrieval.get("source"), "requires_temporal_pair": bool(plan.get("requires_temporal_pair"))}}
