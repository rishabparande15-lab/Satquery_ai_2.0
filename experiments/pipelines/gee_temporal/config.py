from dataclasses import dataclass
from pathlib import Path
import json
import os


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "real_gee" / "gee_temporal"
SAVED_REPORT = OUTPUT_ROOT / "execution_report.json"


@dataclass(frozen=True)
class Config:
    project: str | None
    aoi: dict | None
    before_start: str | None
    before_end: str | None
    after_start: str | None
    after_end: str | None
    output_root: Path = OUTPUT_ROOT


def load_config() -> Config:
    saved = {}
    if SAVED_REPORT.exists():
        try:
            saved = json.loads(SAVED_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
    aoi_text = os.environ.get("GEE_AOI_GEOJSON")
    aoi = json.loads(aoi_text) if aoi_text else saved.get("aoi")
    before_range = saved.get("before_date_range", [None, None])
    after_range = saved.get("after_date_range", [None, None])
    return Config(
        project=os.environ.get("GEE_PROJECT", "satquery-ai-508108" if saved else None),
        aoi=aoi,
        before_start=os.environ.get("GEE_BEFORE_START", before_range[0]),
        before_end=os.environ.get("GEE_BEFORE_END", before_range[1]),
        after_start=os.environ.get("GEE_AFTER_START", after_range[0]),
        after_end=os.environ.get("GEE_AFTER_END", after_range[1]),
    )


def validate_config(config: Config) -> Config:
    missing = []
    if not config.project:
        missing.append("GEE_PROJECT")
    if not config.aoi:
        missing.append("GEE_AOI_GEOJSON")
    for name, value in (
        ("GEE_BEFORE_START", config.before_start),
        ("GEE_BEFORE_END", config.before_end),
        ("GEE_AFTER_START", config.after_start),
        ("GEE_AFTER_END", config.after_end),
    ):
        if not value:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Real-GEE configuration is incomplete. Set these environment variables before running: "
            + ", ".join(missing)
        )
    return config