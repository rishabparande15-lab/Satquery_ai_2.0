import ee
import zipfile
from pathlib import Path

import requests


S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


def _first(collection: ee.ImageCollection, label: str) -> tuple[ee.Image, dict]:
    candidates = collection.aggregate_array("system:id").getInfo() or []
    if not candidates:
        raise RuntimeError(f"No {label} image matched the configured AOI and date range")
    image = ee.Image(sorted(candidates)[0])
    metadata = image.toDictionary(["system:id", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).getInfo()
    if not metadata or not metadata.get("system:id"):
        raise RuntimeError(f"No {label} image matched the configured AOI and date range")
    metadata["label"] = label
    metadata["acquisition_date"] = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    metadata["collection"] = "COPERNICUS/S1_GRD" if "sentinel_1" in label else "COPERNICUS/S2_SR_HARMONIZED"
    metadata["bands"] = image.bandNames().getInfo()
    metadata["projection"] = image.select(0).projection().getInfo()
    return image, metadata


def retrieve(aoi: dict, before: tuple[str, str], after: tuple[str, str]) -> dict:
    geometry = ee.Geometry(aoi)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geometry).filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geometry).filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 30))
    before_s1, before_s1_meta = _first(s1.filterDate(*before), "before_sentinel_1")
    after_s1, after_s1_meta = _first(s1.filterDate(*after), "after_sentinel_1")
    before_s2, before_s2_meta = _first(s2.filterDate(*before), "before_sentinel_2")
    after_s2, after_s2_meta = _first(s2.filterDate(*after), "after_sentinel_2")
    return {"aoi": aoi, "before": {"sentinel_1": before_s1, "sentinel_2": before_s2}, "after": {"sentinel_1": after_s1, "sentinel_2": after_s2}, "metadata": {"sentinel_1": [before_s1_meta, after_s1_meta], "sentinel_2": [before_s2_meta, after_s2_meta]}}


def download_image(image: ee.Image, bands: list[str], aoi: dict, output_root: Path, label: str, scale: int = 10) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    zip_path = output_root / f"{label}.zip"
    response = requests.get(image.getDownloadURL({
        "name": label,
        "bands": bands,
        "region": aoi,
        "scale": scale,
        "filePerBand": True,
        "format": "ZIPPED_GEO_TIFF_PER_BAND",
    }), timeout=180)
    response.raise_for_status()
    zip_path.write_bytes(response.content)
    extract_root = output_root / label
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)
    rasters = sorted(str(path) for path in extract_root.glob("*.tif"))
    if len(rasters) != len(bands):
        raise RuntimeError(f"Expected {len(bands)} GeoTIFFs for {label}, found {len(rasters)}")
    return {"archive": str(zip_path), "rasters": rasters, "bands": bands, "scale": scale}