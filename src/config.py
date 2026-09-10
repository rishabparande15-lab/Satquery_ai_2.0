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
    dataset_root = Path(os.environ.get(
        "DATASET_ROOT",
        r"D:\Satquery_ai datasets\extracted\small-sample",
    ))
    features_root = Path(os.environ.get(
        "FEATURES_ROOT",
        r"D:\Satquery_ai datasets\croma_features",
    ))
    croma_source = Path(os.environ.get(
        "CROMA_SOURCE",
        r"D:\Satquery_ai datasets\croma_official",
    ))
    croma_checkpoint = Path(os.environ.get(
        "CROMA_CHECKPOINT",
        str(features_root.parent / "checkpoints" / "CROMA_base.pt"),
    ))
    return Settings(dataset_root, features_root, croma_source, croma_checkpoint)