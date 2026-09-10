import json
from pathlib import Path

import numpy as np


def inspect_report(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for sample in report.get("samples", []):
        for output in sample.get("output_shapes", {}).values():
            array = np.load(output["path"])
            if list(array.shape) != output["shape"]:
                raise ValueError(f"Shape mismatch in {output['path']}")
    return {"samples": len(report.get("samples", [])), "errors": report.get("errors", []), "valid": not report.get("errors")}
