import os
from pathlib import Path

import numpy as np
import pytest

from src.config import get_settings
from src.dataset_loader import discover_samples, load_sample


DATASET_ROOT = Path(os.environ.get(
    "SATQUERY_TEST_DATASET_ROOT",
    get_settings().dataset_root,
))
pytestmark = pytest.mark.skipif(
    not (DATASET_ROOT / "metadata.parquet").is_file(),
    reason="real three-sample dataset is not configured",
)


def test_real_three_sample_dataset_loads():
    samples = discover_samples(DATASET_ROOT)
    assert [sample.patch_id for sample in samples] == ["61_39", "61_40", "61_41"]
    for sample in samples:
        prepared = load_sample(sample)
        assert prepared.raw_optical.shape == (12, 120, 120)
        assert prepared.raw_sar.shape == (2, 120, 120)
        assert prepared.metadata["crs"] == "EPSG:32633"
        assert prepared.metadata["resolution"] == [10.0, 10.0]
        assert np.isfinite(prepared.optical).all()
        assert np.isfinite(prepared.sar).all()
        assert sample.reference_map.exists()


def test_metadata_labels_are_available():
    import pandas as pd

    frame = pd.read_parquet(DATASET_ROOT / "metadata.parquet")
    assert len(frame) == 3
    assert all(hasattr(labels, "__len__") and len(labels) > 0 for labels in frame["labels"])
