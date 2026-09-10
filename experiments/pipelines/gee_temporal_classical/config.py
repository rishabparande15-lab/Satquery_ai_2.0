from dataclasses import replace
from pathlib import Path

from experiments.pipelines.gee_temporal.config import load_config as load_temporal_config, validate_config


def load_config():
	config = validate_config(load_temporal_config())
	output_root = Path(__file__).resolve().parents[3] / "experiments" / "outputs" / "real_gee" / "gee_temporal_classical"
	return replace(config, output_root=output_root)