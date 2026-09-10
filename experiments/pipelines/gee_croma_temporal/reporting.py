import json
from pathlib import Path


def write_report(report: dict, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "execution_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path