from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from rasterio.transform import from_origin

from src.config import get_settings
from src.dataset_loader import OPTICAL_BANDS, SAR_BANDS
from src.phase1_foundation import prepare_batch
from src.phase2_integration import adapt_phase1_output
from src.phase3_5_end_to_end import run
from src.phase3_orchestration import (
    AOI, FailureCode, Modality, OrchestrationError, SceneCandidate, TemporalRequest,
    validate_raster, validate_spatial_pair,
)


def _real_fixture_available():
    settings = get_settings()
    return settings.dataset_root.exists() and settings.croma_checkpoint.exists() and (settings.croma_source / "use_croma.py").exists()


@pytest.mark.skipif(not _real_fixture_available(), reason="real BigEarthNet/CROMA fixture is unavailable")
def test_real_sample_traverses_phase3_phase1_phase2_and_advanced_pipeline():
    result = run(get_settings().dataset_root, sample_id="61_39", device="cpu", request_id="phase3-5-test-61_39")
    receipt = result.receipt
    assert receipt["status"] == "complete"
    assert receipt["final_verdict"] == "GREEN"
    assert receipt["provenance"]["request"]["request_id"] == "phase3-5-test-61_39"
    assert receipt["provenance"]["phase3"]["status"] == "validated"
    assert receipt["provenance"]["phase1"]["normalization_profile"] == "croma_readme_patch_8bit_v1"
    assert receipt["provenance"]["phase2"]["status"] == "validated"
    assert receipt["provenance"]["advanced_pipeline"]["name"] == "HybridFusion"
    assert receipt["croma_shapes"]["optical_encodings"] == [1, 225, 768]
    assert receipt["croma_shapes"]["SAR_encodings"] == [1, 225, 768]
    assert receipt["croma_shapes"]["joint_encodings"] == [1, 225, 768]
    assert receipt["output_shapes"] == {"physical": [1, 62], "pooled_croma": [1, 2304], "hybrid": [1, 192]}
    assert "61_39" in result.processing_result.selected_scenes[0]
    assert "61_39" in result.processing_result.selected_scenes[1]
    assert np.isfinite(result.physical_features).all() and np.isfinite(result.hybrid_features).all()
    assert result.phase2_input.provenance[0]["identity"]["patch_id"] == "phase3-5-test-61_39"
    assert result.phase2_input.provenance[0]["source_hashes"]["optical_B02"]


@pytest.mark.skipif(not _real_fixture_available(), reason="real BigEarthNet/CROMA fixture is unavailable")
def test_real_execution_repeats_with_exact_feature_hashes():
    first = run(get_settings().dataset_root, sample_id="61_39", device="cpu", request_id="phase3-5-repeat-61_39")
    second = run(get_settings().dataset_root, sample_id="61_39", device="cpu", request_id="phase3-5-repeat-61_39")
    assert first.receipt["feature_hashes"] == second.receipt["feature_hashes"]
    assert first.receipt["provenance"]["acquisition"]["source_hashes"] == second.receipt["provenance"]["acquisition"]["source_hashes"]
    assert np.array_equal(first.physical_features, second.physical_features)
    assert np.array_equal(first.hybrid_features, second.hybrid_features)
    for key in first.croma_features:
        assert torch.equal(first.croma_features[key], second.croma_features[key])


def test_failure_paths_reject_scientifically_invalid_inputs(tmp_path):
    aoi = AOI.from_input({"type": "bbox", "bounds": [0, 0, 1, 1]})
    temporal = TemporalRequest.create(acquisition_date="2026-01-01")
    scene = SceneCandidate("scene", "sentinel-2", "Sentinel-2", "MSI", Modality.OPTICAL,
                           datetime(2026, 1, 1, tzinfo=timezone.utc), 10, "EPSG:4326", (0, 0, 1, 1), 1,
                           tuple(OPTICAL_BANDS))
    path = tmp_path / "invalid.tif"
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=120, width=120, count=11, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(0, 1, 1 / 120, 1 / 120)) as dataset:
        dataset.write(np.ones((11, 120, 120), dtype=np.float32))
        for index, band in enumerate(OPTICAL_BANDS[:-1], 1):
            dataset.set_band_description(index, band)
    with pytest.raises(OrchestrationError) as error:
        validate_raster(path, scene, aoi, Modality.OPTICAL)
    assert error.value.code == FailureCode.INVALID_RASTER

    base = {"crs": "EPSG:32632", "bounds": (0, 0, 1200, 1200), "resolution": (10, 10),
            "transform": (10, 0, 0, 0, -10, 1200, 0, 0, 1), "array": np.ones((12, 120, 120))}
    with pytest.raises(OrchestrationError) as error:
        validate_spatial_pair(base, {**base, "crs": "EPSG:32633", "array": np.ones((2, 120, 120))})
    assert error.value.code == FailureCode.MODALITY_PAIR_INCOMPATIBLE


def test_phase_boundary_rejects_invalid_croma_physical_and_split_contracts():
    from src.phase3_orchestration import AOI, ValidatedSatelliteDataset, bridge_to_phase2

    identity = {"source": "fixture"}
    dataset = ValidatedSatelliteDataset("request", ("s2", "s1"), np.ones((12, 120, 120), dtype=np.float32),
                                       np.ones((2, 120, 120), dtype=np.float32), AOI.from_input({"type": "bbox", "bounds": [0, 0, 1, 1]}),
                                       "EPSG:32632", (0, 0, 1200, 1200), 10, 10,
                                       {"optical": tuple(OPTICAL_BANDS), "sar": tuple(SAR_BANDS)},
                                       ("2026-01-01", "2026-01-01"), {"o": "hash"}, identity)

    class InvalidCroma:
        def __call__(self, *, optical_images, SAR_images):
            z = torch.zeros(len(optical_images), 224, 768)
            return {"optical_encodings": z, "SAR_encodings": z, "joint_encodings": z,
                    "optical_GAP": torch.zeros(len(optical_images), 768), "SAR_GAP": torch.zeros(len(optical_images), 768),
                    "joint_GAP": torch.zeros(len(optical_images), 768)}

    class Physical:
        def extract(self, optical, sar, optical_bands, sar_bands):
            return np.ones(2, dtype=np.float32), {"dimension": 3, "feature_names": ["a", "b", "c"]}

    with pytest.raises(ValueError, match="Invalid CROMA"):
        bridge_to_phase2(dataset, InvalidCroma(), Physical(), croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"})