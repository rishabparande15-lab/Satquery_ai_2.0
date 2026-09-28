import numpy as np
import pytest

from src.eo_vlm_adapter import EOInput, Qwen25VLRGBAdapter, RemoteSensingVisionAdapter, S2_BANDS


def test_capabilities_declare_rgb_only_boundary():
    capabilities = Qwen25VLRGBAdapter().get_capabilities()
    assert capabilities["rgb"] == "PASS"
    assert capabilities["sar_vv_vh"] == "NOT SUPPORTED"
    assert capabilities["hybrid_830d"] == "NOT SUPPORTED"


def test_explicit_input_contracts_preserve_scientific_boundaries():
    contracts = Qwen25VLRGBAdapter().get_input_contracts()
    assert contracts["OPTICAL_RGB"].status == "SUPPORTED"
    assert contracts["S2_MULTISPECTRAL"].status == "ADAPTER_REQUIRED"
    assert contracts["S2_MULTISPECTRAL"].shape == (None, 12, 120, 120)
    assert contracts["S1_SAR"].semantics["bands"] == ("VV", "VH")
    assert contracts["HYBRID_830D"].qwen_path == "SCIENTIFIC_PREDICTOR_ONLY"


def test_optical_sar_and_temporal_are_contracts_not_fake_fusion():
    contracts = Qwen25VLRGBAdapter().get_input_contracts()
    assert contracts["OPTICAL_SAR"].semantics["implemented_fusion"] is False
    assert contracts["TEMPORAL_PAIR"].status == "NOT_IMPLEMENTED"


def test_croma_requires_projector():
    contracts = Qwen25VLRGBAdapter().get_input_contracts()
    assert contracts["CROMA_TOKENS"].qwen_path == "PROJECTOR_REQUIRED"
    assert contracts["CROMA_GAP"].qwen_path == "PROJECTOR_REQUIRED"


def test_grounding_contract_is_fail_closed():
    contract = Qwen25VLRGBAdapter().get_grounding_contract()
    assert contract["status"] == "NOT_VERIFIED"
    assert contract["learned_grounding"] is False
    assert contract["phase2f_mapping"] == "FAIL_CLOSED"


def test_rgb_preparation_preserves_provenance():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    prepared = Qwen25VLRGBAdapter().prepare_inputs(EOInput("OPTICAL_RGB", optical=image))
    assert prepared.model_input.shape == (120, 120, 3)
    assert prepared.provenance["scientific_representation_used"] is False
    assert len(prepared.provenance["input_sha256"]) == 64


@pytest.mark.parametrize("modality", ["SAR", "OPTICAL", "OPTICAL_SAR", "TEMPORAL"])
def test_unsupported_modalities_are_rejected(modality):
    with pytest.raises(ValueError, match="unsupported"):
        Qwen25VLRGBAdapter().validate_inputs(EOInput(modality))


def test_invalid_rgb_shape_is_rejected():
    with pytest.raises(ValueError, match="HxWx3"):
        Qwen25VLRGBAdapter().validate_inputs(EOInput("OPTICAL_RGB", optical=np.zeros((12, 120, 120))))


def test_resource_estimate_is_explicitly_unmeasured():
    estimate = Qwen25VLRGBAdapter().estimate_resources()
    assert estimate["status"] == "ESTIMATE_ONLY"
    assert estimate["estimated_vram_bytes"] == 7520955904
    assert estimate["estimated_image_count"] == 1
    assert estimate["peak_vram_bytes"] is None


def test_model_swappable_base_contract_is_present():
    assert issubclass(Qwen25VLRGBAdapter, RemoteSensingVisionAdapter)


def s2_sample():
    return np.stack([np.full((120, 120), float(index + 1), dtype=np.float32) for index in range(12)])


def test_s2_exact_shape_and_band_order():
    adapter = Qwen25VLRGBAdapter()
    prepared = adapter.prepare_s2(s2_sample(), band_order=S2_BANDS)
    assert prepared.model_input.shape == (120, 120, 3)
    with pytest.raises(ValueError, match="shape"):
        adapter.prepare_s2(np.zeros((11, 120, 120), dtype=np.float32), band_order=S2_BANDS)
    with pytest.raises(ValueError, match="band order"):
        adapter.prepare_s2(s2_sample(), band_order=tuple(reversed(S2_BANDS)))


def test_s2_invalid_values_and_declared_nodata():
    adapter = Qwen25VLRGBAdapter()
    bad = s2_sample(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        adapter.prepare_s2(bad, band_order=S2_BANDS)
    nodata = s2_sample(); nodata[0, 0, 0] = -9999
    prepared = adapter.prepare_s2(nodata, band_order=S2_BANDS, nodata_value=-9999)
    assert prepared.provenance["information_loss"] is True


def test_s2_strategies_are_deterministic_and_explicit():
    adapter = Qwen25VLRGBAdapter(); sample = s2_sample()
    true_color = adapter.prepare_s2(sample, band_order=S2_BANDS, strategy="true_color")
    false_color = adapter.prepare_s2(sample, band_order=S2_BANDS, strategy="false_color")
    grouped = adapter.prepare_s2(sample, band_order=S2_BANDS, strategy="band_group_views")
    assert true_color.provenance["bands_used"] == ["B04", "B03", "B02"]
    assert false_color.model_input.shape == (120, 120, 3)
    assert isinstance(grouped.model_input, tuple) and len(grouped.model_input) == 3
    assert grouped.model_input[0].shape == (120, 120, 3)
    assert adapter.prepare_s2(sample, band_order=S2_BANDS).provenance["preprocessing_fingerprint"] == true_color.provenance["preprocessing_fingerprint"]


def test_s2_provenance_and_language_facing_boundary():
    prepared = Qwen25VLRGBAdapter().prepare_s2(s2_sample(), band_order=S2_BANDS,
                                               source_image_id="area-1", source_reference="s2://area-1")
    assert prepared.provenance["source_image_id"] == "area-1"
    assert prepared.provenance["source_s2_reference"] == "s2://area-1"
    assert prepared.provenance["adapter_id"] == "language_facing_s2_adapter_v1"
    assert prepared.provenance["scientific_representation_used"] is False
