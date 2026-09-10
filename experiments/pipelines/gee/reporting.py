import json
from pathlib import Path


def write_report(report: dict, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "execution_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
