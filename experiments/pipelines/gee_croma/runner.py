import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .config import get_gee_croma_settings
from .croma_adapter import ExperimentalCROMAAdapter
from .data_access import load_local_samples, select_mode
from .preprocessing import infer, validate_inputs
from .reporting import write_report

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated GEE + CROMA experimental pipeline.")
    parser.add_argument("--mode", choices=("auto", "local", "gee"), default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.perf_counter()
    settings = get_gee_croma_settings()
    requested = args.mode or settings.mode
    mode, gee = select_mode(requested)
    report = {"pipeline": "gee_croma_experimental", "mode": mode, "requested_mode": requested, "project_root": str(settings.project_root), "dataset_root": str(settings.dataset_root), "checkpoint": str(settings.checkpoint), "output_root": str(settings.output_root), "sample_ids": list(settings.sample_ids), "gee_status": gee, "device": None, "samples": [], "warnings": [], "errors": [], "limitations": []}
    report["limitations"].extend(["No change percentage or change map is produced; CROMA is used only as a feature extractor.", "These samples are suitable for testing data loading and feature extraction, but they are not sufficient for validating before/after change detection."])
    if mode != "gee":
        report["limitations"].append("No real GEE imagery was retrieved; local BigEarthNet files are controlled references only.")
    try:
        pairs = load_local_samples(settings.dataset_root, settings.sample_ids)
        adapter = ExperimentalCROMAAdapter(settings.croma_source, settings.checkpoint)
        report["device"] = str(adapter.device)
        for sample, item in pairs:
            validation = validate_inputs(item.raw_optical, item.raw_sar)
            outputs = infer(adapter, item.raw_optical, item.raw_sar)
            sample_root = settings.output_root / sample.patch_id; sample_root.mkdir(parents=True, exist_ok=True)
            output_shapes = {}; output_files = []
            for name, tensor in outputs.items():
                array = tensor.detach().cpu().numpy(); path = sample_root / f"{name}.npy"; np.save(path, array); output_files.append(str(path)); output_shapes[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "path": str(path)}
            report["samples"].append({"sample_id": sample.patch_id, "source": "local BigEarthNet reference; not GEE imagery", "input_bands": item.metadata["optical_band_order"] + item.metadata["sar_band_order"], "input_metadata": item.metadata, "validation": validation, "output_shapes": output_shapes, "output_files": output_files, "temporal_metadata_available": False})
    except Exception as error:
        report["errors"].append(str(error))
    report["runtime_seconds"] = time.perf_counter() - started
    report_path = write_report(report, settings.output_root)
    LOGGER.info("report=%s mode=%s device=%s", report_path, mode, report["device"])
    if report["errors"]:
        raise RuntimeError(report["errors"])


if __name__ == "__main__":
    main()
