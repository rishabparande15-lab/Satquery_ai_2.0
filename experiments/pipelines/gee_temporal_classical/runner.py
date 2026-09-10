import argparse
import time

from .config import load_config
from .gee_auth import authenticate, require_authentication
from .reporting import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Real GEE classical temporal change analysis; no local fallback.")
    parser.add_argument("--mode", choices=("gee",), default="gee")
    parser.parse_args()
    config = load_config(); started = time.perf_counter(); auth = authenticate(config.project)
    report = {"data_source": "Google Earth Engine", "fallback_used": False, "authentication_successful": auth["authentication_successful"], "aoi": config.aoi, "before_date_range": [config.before_start, config.before_end], "after_date_range": [config.after_start, config.after_end], "sentinel_1_images": [], "sentinel_2_images": [], "bands": [], "crs": None, "resolution": None, "preprocessing": [], "outputs": [], "runtime_seconds": 0, "warnings": [], "errors": []}
    try:
        require_authentication(config.project)
        if not config.aoi or not all((config.before_start, config.before_end, config.after_start, config.after_end)):
            raise RuntimeError("Set GEE_PROJECT, GEE_AOI_GEOJSON, and both before/after date ranges before retrieval")
        raise RuntimeError("Retrieval and classical analysis are gated until a real authenticated GEE configuration is supplied")
    except Exception as error:
        report["errors"].append(f"Real-GEE classical run stopped before retrieval: {type(error).__name__}: {error}")
        report["warnings"].append("No imagery or change maps were generated and no fallback was used.")
    report["runtime_seconds"] = time.perf_counter() - started
    write_report(report, config.output_root)
    raise RuntimeError(report["errors"][-1])


if __name__ == "__main__":
    main()