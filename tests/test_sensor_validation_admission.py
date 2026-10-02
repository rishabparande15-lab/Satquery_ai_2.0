from src.sensor_validation_admission import (AUTHORIZED_DEVELOPMENT, RESTRICTED_EVALUATION,
                                             classify_authorization, model_validation_gate, product_descriptor)


def test_authorization_requires_explicit_development_split():
    assert classify_authorization({"authorization": AUTHORIZED_DEVELOPMENT, "split": "validation"}) == AUTHORIZED_DEVELOPMENT
    assert classify_authorization({"authorization": AUTHORIZED_DEVELOPMENT, "split": "test"}) == "UNKNOWN"
    assert classify_authorization({"authorization": RESTRICTED_EVALUATION, "split": "test"}) == RESTRICTED_EVALUATION
    assert classify_authorization(None) == "UNKNOWN"


def test_descriptor_preserves_unknown_sensor_metadata_without_invention():
    descriptor = product_descriptor("cartosat-2s", {"band_count": 3, "band_descriptions": ["", "", ""], "dtypes": ["uint16"] * 3,
        "finite_value_checks": {"finite_min": 0.0, "finite_max": 100.0}, "crs": "EPSG:32633", "resolution": [1, 1], "bounds": [0, 0, 1, 1], "nodata": [None] * 3})
    assert descriptor["product_type"] is None and descriptor["status"] == "DESCRIBED_NOT_MODEL_COMPATIBLE"


def test_future_model_gate_remains_closed_for_unknown_or_restricted_data():
    blocked = model_validation_gate(sensor="risat", authorization="UNKNOWN", product_understood=False,
        transformation_defined=False, missing_channels_fabricated=False, domain_mismatch_documented=False, test_access=0)
    assert blocked["model_compatibility_test_authorized"] is False and blocked["model_executed"] is False
    allowed = model_validation_gate(sensor="cartosat-2s", authorization=AUTHORIZED_DEVELOPMENT, product_understood=True,
        transformation_defined=True, missing_channels_fabricated=False, domain_mismatch_documented=True, test_access=0)
    assert allowed["model_compatibility_test_authorized"] is True and allowed["model_executed"] is False
