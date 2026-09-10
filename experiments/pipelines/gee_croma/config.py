from dataclasses import dataclass
from pathlib import Path
import os

from src.config import get_settings


@dataclass(frozen=True)
class GeeCromaSettings:
    project_root: Path
    dataset_root: Path
    checkpoint: Path
    croma_source: Path
    output_root: Path
    sample_ids: tuple[str, ...] = ("61_39", "61_40", "61_41")
    mode: str = "auto"


def get_gee_croma_settings() -> GeeCromaSettings:
    local = get_settings()
    return GeeCromaSettings(
        project_root=Path(__file__).resolve().parents[3],
        dataset_root=local.dataset_root,
        checkpoint=local.croma_checkpoint,
        croma_source=local.croma_source,
        output_root=Path(os.environ.get("GEE_CROMA_OUTPUT_ROOT", str(Path(__file__).resolve().parent / "outputs"))),
        mode=os.environ.get("GEE_CROMA_PIPELINE_MODE", "auto"),
    )
