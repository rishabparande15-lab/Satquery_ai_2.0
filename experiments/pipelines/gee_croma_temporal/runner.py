import argparse
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .croma_adapter import load_adapter
from .data_access import load_pipeline2_outputs
from .gee_auth import authenticate, require_authentication
from .preprocessing import load_croma_inputs
from .reporting import write_report
from .temporal_features import compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Real GEE + CROMA temporal analysis; no local fallback.")
    parser.add_argument("--mode", choices=("gee",), default="gee")
    parser.parse_args()
    config = load_config(); started = time.perf_counter(); auth = authenticate(config.gee.project)
    report = {"data_source": "Google Earth Engine", "fallback_used": False, "authentication_successful": auth["authentication_successful"], "aoi": config.gee.aoi, "before_date_range": [config.gee.before_start, config.gee.before_end], "after_date_range": [config.gee.after_start, config.gee.after_end], "sentinel_1_images": [], "sentinel_2_images": [], "bands": [], "crs": None, "resolution": None, "preprocessing": ["Official CROMA adapter will normalize raw channels per channel using mean +/- 2 std clipped to [0,1]."], "outputs": [], "runtime_seconds": 0, "warnings": [], "errors": [], "croma_checkpoint": str(config.checkpoint), "croma_output_shapes": {}}
    try:
        require_authentication(config.gee.project)
        project_root = Path(__file__).resolve().parents[3]
        pipeline2_report, output_paths = load_pipeline2_outputs(project_root)
        adapter = load_adapter(config.croma_source, config.checkpoint)
        report["device"] = str(adapter.device)
        report["sentinel_1_images"] = pipeline2_report["sentinel_1_images"]
        report["sentinel_2_images"] = pipeline2_report["sentinel_2_images"]
        report["bands"] = pipeline2_report["bands"]
        report["crs"] = pipeline2_report["crs"]
        report["resolution"] = pipeline2_report["resolution"]
        report["preprocessing"].extend(["Consumed only Pipeline 2 real-GEE GeoTIFF outputs.", "Each band was bilinearly resized to 120x120 inside this isolated CROMA pipeline."])
        before_optical, before_sar = load_croma_inputs({"sentinel_1": output_paths["before_sentinel_1"], "sentinel_2": output_paths["before_sentinel_2"]})
        after_optical, after_sar = load_croma_inputs({"sentinel_1": output_paths["after_sentinel_1"], "sentinel_2": output_paths["after_sentinel_2"]})
        before_outputs = adapter.infer(before_optical, before_sar)
        after_outputs = adapter.infer(after_optical, after_sar)
        report["croma_input_shapes"] = {"before_optical": list(before_optical.shape), "before_sar": list(before_sar.shape), "after_optical": list(after_optical.shape), "after_sar": list(after_sar.shape)}
        for period, outputs in (("before", before_outputs), ("after", after_outputs)):
            period_root = config.output_root / period
            period_root.mkdir(parents=True, exist_ok=True)
            for name, tensor in outputs.items():
                array = tensor.detach().cpu().numpy()
                path = period_root / f"{name}.npy"
                np.save(path, array)
                report["outputs"].append({"period": period, "name": name, "shape": list(array.shape), "path": str(path)})
                report["croma_output_shapes"].setdefault(name, {})[period] = list(array.shape)
        report["temporal_comparison"] = compare(before_outputs, after_outputs)
        report["warnings"].append("CROMA representations are not calibrated change percentages or change maps.")
    except Exception as error:
        report["errors"].append(f"Real-GEE + CROMA run stopped before retrieval: {type(error).__name__}: {error}")
        report["warnings"].append("No imagery or CROMA features were generated and no fallback was used.")
    report["runtime_seconds"] = time.perf_counter() - started
    write_report(report, config.output_root)
    if report["errors"]:
        raise RuntimeError(report["errors"][-1])


if __name__ == "__main__":
    main()