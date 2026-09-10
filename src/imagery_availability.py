"""Provider-neutral satellite source registry and imagery discovery."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import importlib.util
import math
import os
from typing import Any


@dataclass(frozen=True)
class SourceDefinition:
    source: str
    name: str
    sensor_type: str
    modality: str
    native_resolution_m: float | None
    bands: tuple[str, ...]
    supported_analysis_types: tuple[str, ...]
    date_coverage: str | None
    integration_status: str
    configuration: str | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bands"] = list(self.bands)
        result["supported_analysis_types"] = list(self.supported_analysis_types)
        return result


SOURCE_REGISTRY = {
    "sentinel-2": SourceDefinition(
        "sentinel-2", "Sentinel-2", "Multispectral optical", "optical", 10.0,
        ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"),
        ("optical analysis", "land-cover features", "water features", "change analysis"),
        "Not fixed in this service; filtered by the requested dates", "requires configuration", "GEE_PROJECT and Earth Engine authentication",
    ),
    "sentinel-1": SourceDefinition(
        "sentinel-1", "Sentinel-1", "Synthetic aperture radar", "sar", 10.0,
        ("VV", "VH"),
        ("SAR analysis", "change analysis", "joint optical-SAR analysis"),
        "Not fixed in this service; filtered by the requested dates", "requires configuration", "GEE_PROJECT and Earth Engine authentication",
    ),
    "landsat": SourceDefinition(
        "landsat", "Landsat", "Multispectral optical", "optical", 30.0,
        (), ("optical analysis", "land-cover features", "water features", "change analysis"),
        None, "planned", "A web imagery provider is not implemented for this source",
    ),
    "cartosat-2s": SourceDefinition(
        "cartosat-2s", "Cartosat-2S", "High-resolution optical", "optical", None,
        (), ("optical analysis", "change analysis"), None, "unavailable", "No verified web integration is configured",
    ),
    "risat": SourceDefinition(
        "risat", "RISAT", "Synthetic aperture radar", "sar", None,
        (), ("SAR analysis", "change analysis"), None, "unavailable", "No verified web integration is configured",
    ),
}

MAX_AOI_SPAN_DEGREES = 20.0
MAX_AOI_AREA_KM2 = 2_000_000.0


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _date(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is required and must be an ISO date.")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError(f"{field} must be a valid ISO date.") from None


def _area_km2(bounds: list[float]) -> float:
    west, south, east, north = bounds
    lat_km = 111.32
    lon_km = 111.32 * max(math.cos(math.radians((south + north) / 2)), 0.01)
    return abs(east - west) * lon_km * abs(north - south) * lat_km


def _bounds_from_coordinates(coordinates: list[list[float]]) -> list[float]:
    points = [point for point in coordinates if isinstance(point, list) and len(point) >= 2]
    if not points or any(not _number(point[0]) or not _number(point[1]) for point in points):
        raise ValueError("AOI polygon coordinates are invalid.")
    longitudes, latitudes = zip(*((point[0], point[1]) for point in points))
    return [min(longitudes), min(latitudes), max(longitudes), max(latitudes)]


def normalize_aoi(aoi: dict[str, Any], source: str = "api") -> dict[str, Any]:
    if not isinstance(aoi, dict):
        raise ValueError("AOI is required and must be a point, bbox, or GeoJSON polygon.")
    kind = aoi.get("type")
    if kind == "point":
        latitude, longitude = aoi.get("latitude"), aoi.get("longitude")
        if not (_number(latitude) and _number(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("AOI point latitude/longitude is invalid.")
        bounds = [longitude, latitude, longitude, latitude]
        geometry = {"type": "Point", "coordinates": [longitude, latitude]}
        area = 0.0
        centroid = {"latitude": latitude, "longitude": longitude}
    elif kind == "bbox":
        bounds = aoi.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 4 or not all(_number(item) for item in bounds):
            raise ValueError("AOI bbox must be [west, south, east, north].")
        west, south, east, north = bounds
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("AOI bbox coordinates are outside supported bounds or empty.")
        geometry = {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}
        area = _area_km2(bounds)
        centroid = {"latitude": (south + north) / 2, "longitude": (west + east) / 2}
    elif kind in ("Polygon", "polygon"):
        coordinates = aoi.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 1 or not isinstance(coordinates[0], list) or len(coordinates[0]) < 4:
            raise ValueError("AOI polygon must contain one closed ring with at least four coordinates.")
        ring = coordinates[0]
        if ring[0] != ring[-1]:
            raise ValueError("AOI polygon ring must be closed; the first and last coordinates must match.")
        bounds = _bounds_from_coordinates(ring)
        if not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
            raise ValueError("AOI polygon coordinates are outside supported bounds.")
        geometry = {"type": "Polygon", "coordinates": [ring]}
        area = _area_km2(bounds)
        centroid = {"latitude": sum(point[1] for point in ring[:-1]) / (len(ring) - 1), "longitude": sum(point[0] for point in ring[:-1]) / (len(ring) - 1)}
    else:
        raise ValueError("AOI type must be point, bbox, or GeoJSON polygon.")
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if span > MAX_AOI_SPAN_DEGREES or area > MAX_AOI_AREA_KM2:
        raise ValueError("AOI is too large for imagery discovery; select a smaller region.")
    return {"geometry": geometry, "bbox": bounds, "centroid": centroid, "area_km2": round(area, 3), "crs": "EPSG:4326", "source": source}


def _resolution_status(requested: float, native: float | None) -> tuple[str, str]:
    if native is None:
        return "unavailable", "Native resolution is not verified for this source."
    if requested < native / 2:
        return "unavailable", f"Requested {requested:g} m, but available imagery is approximately {native:g} m."
    if requested < native:
        return "reduced_detail", f"Requested {requested:g} m, but available imagery is approximately {native:g} m."
    if requested == native:
        return "compatible", f"Requested {requested:g} m. Available imagery supports approximately {native:g} m."
    return "compatible", f"Requested {requested:g} m. Native imagery is approximately {native:g} m; output may be resampled."


def validate_search_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("Availability request must be a JSON object.")
    aoi = normalize_aoi(request.get("aoi"), request.get("aoi_source", "api"))
    start_date = _date(request.get("start_date"), "start_date")
    end_date = _date(request.get("end_date"), "end_date")
    if start_date > end_date:
        raise ValueError("start_date must not be later than end_date.")
    requested = request.get("requested_resolution_m")
    if not _number(requested) or requested <= 0 or requested > 1000:
        raise ValueError("requested_resolution_m must be a finite value between 0 and 1000 metres.")
    modality = request.get("modality", "optical")
    if modality not in {"optical", "sar", "joint"}:
        raise ValueError("modality must be optical, sar, or joint.")
    source = request.get("source")
    if source is not None and source not in SOURCE_REGISTRY:
        raise ValueError("Unknown satellite source.")
    cloud_cover = request.get("cloud_cover")
    if cloud_cover is not None and (not _number(cloud_cover) or not 0 <= cloud_cover <= 100):
        raise ValueError("cloud_cover must be between 0 and 100 percent.")
    sources = [source] if source else (["sentinel-2"] if modality == "optical" else ["sentinel-1"] if modality == "sar" else ["sentinel-2", "sentinel-1"])
    return {"aoi": aoi, "start_date": start_date, "end_date": end_date, "requested_resolution_m": float(requested), "modality": modality, "sources": sources, "cloud_cover": cloud_cover}


def registry_response() -> list[dict[str, Any]]:
    return [source.to_dict() for source in SOURCE_REGISTRY.values()]


def provider_status() -> dict[str, Any]:
    project = os.environ.get("GEE_PROJECT")
    if not project:
        return {"provider": "google-earth-engine", "status": "requires_configuration", "message": "Set GEE_PROJECT and authenticate Earth Engine before searching imagery."}
    if importlib.util.find_spec("ee") is None:
        return {"provider": "google-earth-engine", "status": "requires_configuration", "message": "Install earthengine-api and authenticate Earth Engine before searching imagery."}
    return {"provider": "google-earth-engine", "status": "configured", "message": "Earth Engine credentials will be checked when imagery is searched."}


def _unavailable_response(search: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    options = []
    for key in search["sources"]:
        definition = SOURCE_REGISTRY[key]
        resolution, reason = _resolution_status(search["requested_resolution_m"], definition.native_resolution_m)
        options.append({"source": key, "name": definition.name, "status": status["status"], "coverage_status": "not_checked", "compatibility_status": resolution, "native_resolution_m": definition.native_resolution_m, "reason": status["message"] if status["status"] != "configured" else reason, "acquisitions": []})
    return {"status": status["status"], "provider": status["provider"], "aoi": search["aoi"], "date_range": {"start": search["start_date"], "end": search["end_date"]}, "requested_resolution_m": search["requested_resolution_m"], "options": options, "warnings": [status["message"]]}


def search_imagery(request: dict[str, Any]) -> dict[str, Any]:
    search = validate_search_request(request)
    status = provider_status()
    if status["status"] != "configured":
        return _unavailable_response(search, status)
    try:
        import ee
        ee.Initialize(project=os.environ["GEE_PROJECT"])
    except Exception as error:
        auth_status = {"provider": "google-earth-engine", "status": "requires_configuration", "message": f"Earth Engine authentication is unavailable: {type(error).__name__}."}
        return _unavailable_response(search, auth_status)
    options = []
    geometry = ee.Geometry(search["aoi"]["geometry"])
    for key in search["sources"]:
        definition = SOURCE_REGISTRY[key]
        if key == "sentinel-2":
            collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geometry).filterDate(search["start_date"], search["end_date"])
            if search["cloud_cover"] is not None:
                collection = collection.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", search["cloud_cover"]))
        elif key == "sentinel-1":
            collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geometry).filterDate(search["start_date"], search["end_date"]).filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        else:
            continue
        try:
            items = collection.sort("system:time_start").limit(20).toList(20).getInfo() or []
            acquisitions = []
            for item in items:
                properties = item.get("properties", {})
                timestamp = properties.get("system:time_start")
                acquisitions.append({"id": item.get("id"), "acquisition_date": date.fromtimestamp(timestamp / 1000).isoformat() if timestamp else None, "cloud_percentage": properties.get("CLOUDY_PIXEL_PERCENTAGE") if key == "sentinel-2" else None, "bands": item.get("bands", [])})
            compatibility, reason = _resolution_status(search["requested_resolution_m"], definition.native_resolution_m)
            options.append({"source": key, "name": definition.name, "status": "available" if acquisitions and compatibility != "unavailable" else "unavailable", "coverage_status": "found" if acquisitions else "no_acquisition", "compatibility_status": compatibility, "native_resolution_m": definition.native_resolution_m, "reason": reason if acquisitions else "No acquisition matched the AOI and date range.", "acquisitions": acquisitions})
        except Exception:
            options.append({"source": key, "name": definition.name, "status": "unavailable", "coverage_status": "query_failed", "compatibility_status": "unknown", "native_resolution_m": definition.native_resolution_m, "reason": "The provider query failed; no imagery result is asserted.", "acquisitions": []})
    return {"status": "available" if any(option["status"] == "available" for option in options) else "unavailable", "provider": "google-earth-engine", "aoi": search["aoi"], "date_range": {"start": search["start_date"], "end": search["end_date"]}, "requested_resolution_m": search["requested_resolution_m"], "options": options, "warnings": []}
