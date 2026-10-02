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
    levir_root: Path
    s2_projector_checkpoint: Path
    joint_projector_checkpoint: Path
    chg2cap_checkpoint: Path
    scene_description_model: Path | None
    host: str
    port: int
    device_preference: str
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
    return Settings(
        dataset_root, features_root, croma_source, croma_checkpoint,
        Path(os.environ.get("LEVIR_ROOT", str(PROJECT_ROOT / "datasets" / "levir_cc" / "development_smoke"))),
        Path(os.environ.get("S2_PROJECTOR_CHECKPOINT", str(PROJECT_ROOT / "artifacts" / "training" / "phase3o" / "phase3o5_run" / "s2_multispectral_projector.pt"))),
        Path(os.environ.get("JOINT_PROJECTOR_CHECKPOINT", str(PROJECT_ROOT / "artifacts" / "training" / "phase3q" / "phase3q1_fusion_adaptation" / "phase3q1_joint_projector_final.pt"))),
        Path(os.environ.get("CHG2CAP_CHECKPOINT", str(PROJECT_ROOT / "checkpoints" / "chg2cap" / "LEVIR_CC_batchsize_32_resnet101.pth"))),
        Path(os.environ["SATQUERY_SCENE_DESCRIPTION_MODEL"]) if os.environ.get("SATQUERY_SCENE_DESCRIPTION_MODEL") else None,
        os.environ.get("SATQUERY_HOST", "127.0.0.1"), int(os.environ.get("SATQUERY_PORT", "8000")),
        os.environ.get("SATQUERY_DEVICE", "cuda"),
    )
