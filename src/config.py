from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    dataset_root: Path
    features_root: Path
    croma_source: Path
    croma_checkpoint: Path
    croma_size: str = "base"
    croma_image_resolution: int = 120


def get_settings() -> Settings:
    local_data_root = Path(os.environ.get(
        "SATQUERY_LOCAL_DATA_ROOT",
        str(PROJECT_ROOT / "data" / "raw"),
    ))
    dataset_root = Path(os.environ.get(
        "DATASET_ROOT",
        str(local_data_root / "bigearthnet-v2-small-sample"),
    ))
    features_root = Path(os.environ.get(
        "FEATURES_ROOT",
        str(local_data_root / "croma-features"),
    ))
    croma_source = Path(os.environ.get(
        "CROMA_SOURCE",
        str(local_data_root / "croma-official"),
    ))
    croma_checkpoint = Path(os.environ.get(
        "CROMA_CHECKPOINT",
        str(local_data_root / "checkpoints" / "CROMA_base.pt"),
    ))
    return Settings(dataset_root, features_root, croma_source, croma_checkpoint)
