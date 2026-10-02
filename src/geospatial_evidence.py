"""Measured, fail-closed geospatial evidence for SatQuery responses.

This module reports only uploaded-raster facts and route provenance.  It does
not locate objects, infer alignment, or turn pixel differences into change maps.
"""
from __future__ import annotations
from typing import Any


def _valid_bounds(bounds: Any) -> bool:
    return isinstance(bounds, (list, tuple)) and len(bounds) == 4 and all(isinstance(v, (int, float)) for v in bounds) and bounds[0] < bounds[2] and bounds[1] < bounds[3]


def raster_evidence(input_id: str, inspection: dict[str, Any] | None, *, declaration=None, adapter=None, compatibility=None) -> dict[str, Any]:
    inspection = inspection or {}
    crs, bounds = inspection.get("crs"), inspection.get("bounds")
    available = bool(crs and _valid_bounds(bounds))
    footprint = {"status": "AVAILABLE", "crs": crs, "bounds": list(bounds), "geometry": "BOUNDING_RECTANGLE"} if available else {"status": "GEOSPATIAL_FOOTPRINT_NOT_AVAILABLE", "crs": crs, "bounds": None, "geometry": None}
    if available:
        try:
            from rasterio.warp import transform_bounds
            footprint["wgs84_bounds"] = list(transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21))
        except Exception:
            footprint["wgs84_bounds"] = None
            footprint["wgs84_status"] = "WGS84_TRANSFORM_NOT_AVAILABLE"
    return {"input_id": str(input_id), "geospatial": {"available": available, "crs": crs,
            "bounds": list(bounds) if _valid_bounds(bounds) else None, "transform": inspection.get("transform"),
            "resolution": inspection.get("resolution"), "width": inspection.get("width"), "height": inspection.get("height"), "footprint": footprint},
            "raster_statistics": {"label": "IMAGE STATISTICS", "bands": inspection.get("band_statistics", []),
            "band_count": inspection.get("band_count"), "band_names": inspection.get("band_descriptions"), "dtypes": inspection.get("dtypes"), "nodata": inspection.get("nodata"),
            "finite_value_checks": inspection.get("finite_value_checks")},
            "source": {"file_name": inspection.get("file_name"), "sha256": inspection.get("sha256"), "driver": inspection.get("driver")},
            "declaration": declaration, "adapter": adapter, "compatibility": compatibility,
            "warnings": ([] if available else ["GEOSPATIAL_FOOTPRINT_NOT_AVAILABLE"])}


def pair_evidence(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    a, b = first["geospatial"], second["geospatial"]
    base = {"label": "SPATIAL BOUNDS OVERLAP", "coregistration": "COREGISTRATION_NOT_VERIFIED", "pixel_difference": {"status": "PIXEL_DIFFERENCE_NOT_AVAILABLE", "label": "NON-SEMANTIC PIXEL DIFFERENCE", "reason": "Alignment was not established by this route."}}
    if not a["available"] or not b["available"]:
        return {**base, "status": "GEOSPATIAL_FOOTPRINT_NOT_AVAILABLE", "intersection": None}
    if a["crs"] != b["crs"]:
        return {**base, "status": "PAIR_CRS_MISMATCH", "intersection": None}
    ax0, ay0, ax1, ay1 = a["bounds"]; bx0, by0, bx1, by1 = b["bounds"]
    hit = [max(ax0,bx0), max(ay0,by0), min(ax1,bx1), min(ay1,by1)]
    area = max(0, hit[2]-hit[0]) * max(0, hit[3]-hit[1])
    area_a, area_b = (ax1-ax0)*(ay1-ay0), (bx1-bx0)*(by1-by0)
    return {**base, "status": "OVERLAP" if area else "NO_OVERLAP", "intersection": {"bounds": hit if area else None, "area": area, "fraction_of_first": area / area_a if area_a else None, "fraction_of_second": area / area_b if area_b else None}}


def response_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional executor-supplied inputs into safe JSON evidence."""
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    raw = details.get("evidence_inputs", []) if isinstance(details, dict) else []
    inputs = [raster_evidence(item.get("input_id", "input"), item.get("inspection"), declaration=item.get("declaration"), adapter=item.get("adapter"), compatibility=item.get("compatibility")) for item in raw if isinstance(item, dict)]
    evidence = {"schema": "SATQUERY_GEOSPATIAL_EVIDENCE_V1", "inputs": inputs,
                "provenance": {"route": result.get("route"), "specialist": (result.get("provenance") or {}).get("specialist"), "warnings": result.get("warnings", []), "confidence_policy": "MODEL_CONFIDENCE_IS_SEPARATE_FROM_IMAGE_STATISTICS"}}
    if len(inputs) == 2: evidence["pair"] = pair_evidence(inputs[0], inputs[1])
    return evidence
