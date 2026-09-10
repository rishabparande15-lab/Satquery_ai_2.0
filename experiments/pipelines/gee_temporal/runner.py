import argparse
import time
import json

import numpy as np
import rasterio

from .config import load_config, validate_config
from .gee_auth import authenticate, require_authentication
from .reporting import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Real GEE temporal analysis; no local fallback.")
    parser.add_argument("--mode", choices=("gee",), default="gee")
    args = parser.parse_args()
    config = load_config(); started = time.perf_counter()
    try:
        validate_config(config)
        auth = authenticate(config.project)
    except Exception as error:
        auth = {"authentication_successful": False, "project": config.project, "error": str(error)}
    report = {"data_source": "Google Earth Engine", "fallback_used": False, "authentication_successful": auth["authentication_successful"], "aoi": config.aoi, "before_date_range": [config.before_start, config.before_end], "after_date_range": [config.after_start, config.after_end], "sentinel_1_images": [], "sentinel_2_images": [], "bands": [], "crs": None, "resolution": None, "preprocessing": [], "outputs": [], "runtime_seconds": 0, "warnings": [], "errors": []}
    try:
        validate_config(config)
        require_authentication(config.project)
        from .data_access import S1_BANDS, S2_BANDS, download_image, retrieve
        result = retrieve(config.aoi, (config.before_start, config.before_end), (config.after_start, config.after_end))
        report["bands"] = {"sentinel_1": S1_BANDS, "sentinel_2": S2_BANDS}
        report["sentinel_1_images"] = result["metadata"]["sentinel_1"]
        report["sentinel_2_images"] = result["metadata"]["sentinel_2"]
        report["preprocessing"] = ["Sentinel-1 VV/VH and Sentinel-2 bands downloaded from GEE at 10 m scale.", "Per-band GeoTIFFs retained without local fallback."]
        for period in ("before", "after"):
            for sensor, bands in (("sentinel_1", S1_BANDS), ("sentinel_2", S2_BANDS)):
                report["outputs"].append(download_image(result[period][sensor], bands, config.aoi, config.output_root, f"{period}_{sensor}"))
        report["crs"] = report["sentinel_2_images"][0]["projection"].get("crs")
        report["resolution"] = 10
        for output in report["outputs"]:
            output["rasters_metadata"] = []
            for raster_path in output["rasters"]:
                with rasterio.open(raster_path) as dataset:
                    output["rasters_metadata"].append({"path": raster_path, "width": dataset.width, "height": dataset.height, "dtype": dataset.dtypes[0], "crs": str(dataset.crs), "resolution": list(dataset.res)})
        before_s2 = {path.split(".")[-2]: path for path in report["outputs"][1]["rasters"]}
        after_s2 = {path.split(".")[-2]: path for path in report["outputs"][3]["rasters"]}
        stats = {"method": "absolute before/after raster statistics; no calibrated change percentage", "bands": {}}
        for band in S2_BANDS:
            with rasterio.open(before_s2[band]) as before_dataset, rasterio.open(after_s2[band]) as after_dataset:
                before_array = before_dataset.read(1).astype(np.float32)
                after_array = after_dataset.read(1).astype(np.float32)
                difference = after_array - before_array
                stats["bands"][band] = {"before_mean": float(np.nanmean(before_array)), "after_mean": float(np.nanmean(after_array)), "difference_mean": float(np.nanmean(difference)), "difference_abs_mean": float(np.nanmean(np.abs(difference)))}
        stats_path = config.output_root / "temporal_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        report["outputs"].append({"temporal_statistics": str(stats_path)})
    except Exception as error:
        report["errors"].append(f"Real-GEE run stopped before retrieval: {type(error).__name__}: {error}")
        report["warnings"].append("No imagery was retrieved and no fallback was used.")
    report["runtime_seconds"] = time.perf_counter() - started
    write_report(report, config.output_root)
    if report["errors"]:
        raise RuntimeError(report["errors"][-1])


if __name__ == "__main__":
    main()