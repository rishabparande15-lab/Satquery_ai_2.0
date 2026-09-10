import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import get_gee_settings
from .data_access import load_local_references, local_summary, select_mode
from .preprocessing import normalized_display
from .reporting import write_report

LOGGER = logging.getLogger(__name__)


def save_local_visualizations(reference, output_root: Path) -> list[str]:
    item = reference.prepared
    sample_root = output_root / reference.sample.patch_id
    sample_root.mkdir(parents=True, exist_ok=True)
    paths = []
    rgb = np.stack([normalized_display(item.raw_optical[index]) for index in (3, 2, 1)], axis=-1)
    figure, axis = plt.subplots(figsize=(5, 5)); axis.imshow(rgb); axis.set_title("Local reference RGB (not GEE)"); axis.axis("off")
    path = sample_root / "local_reference_rgb.png"; figure.savefig(path, dpi=120, bbox_inches="tight"); plt.close(figure); paths.append(str(path))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, item.raw_sar, ("VV", "VH", "VV - VH")):
        shown = normalized_display(image if title != "VV - VH" else item.raw_sar[0] - item.raw_sar[1]); axis.imshow(shown, cmap="magma"); axis.set_title(title); axis.axis("off")
    path = sample_root / "local_reference_sar.png"; figure.savefig(path, dpi=120, bbox_inches="tight"); plt.close(figure); paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated GEE-only experimental pipeline.")
    parser.add_argument("--mode", choices=("auto", "local", "gee"), default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.perf_counter()
    settings = get_gee_settings()
    requested = args.mode or settings.mode
    mode, gee = select_mode(requested)
    report = {"pipeline": "gee_only_experimental", "mode": mode, "requested_mode": requested, "project_root": str(settings.project_root), "dataset_root": str(settings.dataset_root), "output_root": str(settings.output_root), "sample_ids": list(settings.sample_ids), "gee_status": gee, "samples": [], "warnings": [], "errors": [], "limitations": []}
    if mode != "gee":
        report["limitations"].extend(["No real GEE imagery was retrieved.", "Local files are controlled references only and are not GEE outputs.", "The samples lack sufficient temporal metadata for before/after change validation.", "These samples are suitable for testing data loading and feature extraction, but they are not sufficient for validating before/after change detection."])
        try:
            references = load_local_references(settings.dataset_root, settings.sample_ids)
            for reference in references:
                summary = local_summary(reference); summary["output_files"] = save_local_visualizations(reference, settings.output_root); report["samples"].append(summary)
        except Exception as error:
            report["errors"].append(str(error))
    else:
        report["limitations"].append("Authenticated GEE mode is configured, but no area/date was supplied; no invented query was made.")
    report["runtime_seconds"] = time.perf_counter() - started
    report_path = write_report(report, settings.output_root)
    LOGGER.info("report=%s mode=%s", report_path, mode)
    if report["errors"]:
        raise RuntimeError(report["errors"])


if __name__ == "__main__":
    main()
