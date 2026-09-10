from dataclasses import dataclass
from pathlib import Path
import os

from src.config import get_settings


@dataclass(frozen=True)
class GeeSettings:
    project_root: Path
    dataset_root: Path
    output_root: Path
    sample_ids: tuple[str, ...] = ("61_39", "61_40", "61_41")
    mode: str = "auto"


def get_gee_settings() -> GeeSettings:
    local = get_settings()
    return GeeSettings(
        project_root=Path(__file__).resolve().parents[3],
        dataset_root=local.dataset_root,
        output_root=Path(os.environ.get("GEE_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "outputs"))),
        mode=os.environ.get("GEE_PIPELINE_MODE", "auto"),
    )
