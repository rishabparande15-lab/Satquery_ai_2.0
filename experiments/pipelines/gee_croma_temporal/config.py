from dataclasses import dataclass
from pathlib import Path
import os
from experiments.pipelines.gee_temporal.config import load_config as load_gee_config, validate_config
from src.config import get_settings


@dataclass(frozen=True)
class Config:
    gee: object
    croma_source: Path
    checkpoint: Path
    output_root: Path


def load_config() -> Config:
    local = get_settings()
    return Config(validate_config(load_gee_config()), local.croma_source, local.croma_checkpoint, Path(__file__).resolve().parents[3] / "experiments" / "outputs" / "real_gee" / "gee_croma_temporal")