from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.interpretation_adapter import (
    UNAVAILABLE, interpret_evidence, technical_companion, validate_interpretation,
)


EVIDENCE_PATH = Path(__file__).parents[1] / "artifacts" / "spatial_evidence" / "61_39" / "evidence.json"


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("question", "claim_id", "sensor"), [
    ("What vegetation evidence is present?", "vegetation", "OPTICAL"),
    ("What water evidence is present?", "water", "OPTICAL"),
    ("What does SAR show?", "sar_polarization", "SAR"),
])
def test_supported_sensor_questions(evidence, question, claim_id, sensor):
    result=interpret_evidence(evidence,question)
    assert result["status"] == "ANSWERED"
    assert [claim["claim_id"] for claim in result["claims"]] == [claim_id]
    assert result["claims"][0]["sensor"] == sensor
    validate_interpretation(result)


def test_scene_overview_keeps_optical_and_sar_distinct(evidence):
    result=interpret_evidence(evidence,"What is present in this image?")
    assert {claim["sensor"] for claim in result["claims"]} == {"OPTICAL","SAR"}
    assert "not a joint semantic prediction" in result["answer"]


def test_where_question_uses_only_geometry_derived_position(evidence):
    result=interpret_evidence(evidence,"Where is vegetation located?")
    assert "deterministic evidence regions" in result["answer"]
    assert result["claims"][0]["region_positions"]


def test_no_geometry_means_no_location_language(evidence):
    modified=deepcopy(evidence)
    next(c for c in modified["claims"] if c["claim_id"]=="vegetation")["region_ids"]=[]
    result=interpret_evidence(modified,"Where is vegetation located?")
    assert result["status"] == "ANSWERED"
    assert "portion" not in result["answer"] and "regions" not in result["answer"]


def test_technical_mode_exposes_feature_tokens_regions_and_calibration(evidence):
    result=technical_companion(evidence,"Give technical details about water evidence")
    assert "Supporting feature: NDWI" in result["answer"]
    assert "tokens:" in result["answer"] and "region IDs:" in result["answer"]
    assert "not calibrated confidence" in result["answer"]


def test_simple_mode_does_not_dump_token_details(evidence):
    result=interpret_evidence(evidence,"What water evidence is present?",mode="simple")
    assert "Supporting feature:" not in result["answer"]


@pytest.mark.parametrize("question", [
    "How many houses are present?", "Locate every car", "Is there a road?",
    "What company owns this site?", "Give the exact identity", "What is the exact address?",
])
def test_unsupported_object_count_and_identity_questions_fail_closed(evidence, question):
    result=interpret_evidence(evidence,question)
    assert result["status"] == "UNAVAILABLE"
    assert result["answer"] == UNAVAILABLE and result["claims"] == []
    validate_interpretation(result)


def test_unrouted_question_fails_closed(evidence):
    assert interpret_evidence(evidence,"Who lives here?")["status"] == "UNAVAILABLE"


def test_missing_evidence_fails_closed():
    result=interpret_evidence(None,"What water evidence is present?")
    assert result["status"] == "UNAVAILABLE" and not result["claims"]


def test_missing_sensor_has_no_cross_sensor_fallback(evidence):
    modified=deepcopy(evidence)
    modified["claims"]=[c for c in modified["claims"] if c["sensor_view"]!="SAR"]
    result=interpret_evidence(modified,"What does SAR show?")
    assert result["status"] == "UNAVAILABLE"


def test_optical_missing_returns_sar_only(evidence):
    modified=deepcopy(evidence)
    modified["claims"]=[c for c in modified["claims"] if c["sensor_view"]=="SAR"]
    result=interpret_evidence(modified,"What is present in this image?")
    assert result["status"] == "ANSWERED"
    assert {c["sensor"] for c in result["claims"]} == {"SAR"}


def test_sar_missing_returns_optical_only(evidence):
    modified=deepcopy(evidence)
    modified["claims"]=[c for c in modified["claims"] if c["sensor_view"]=="OPTICAL"]
    result=interpret_evidence(modified,"What is present in this image?")
    assert result["status"] == "ANSWERED"
    assert {c["sensor"] for c in result["claims"]} == {"OPTICAL"}


def test_non_allowlisted_built_up_claim_is_rejected(evidence):
    modified=deepcopy(evidence)
    unsupported=deepcopy(next(c for c in modified["claims"] if c["claim_id"]=="water"))
    unsupported["claim_id"]="built_up"
    modified["claims"].append(unsupported)
    result=interpret_evidence(modified,"What evidence is present?")
    assert "built_up" not in [c["claim_id"] for c in result["claims"]]
    assert any(c["claim_id"]=="built_up" for c in result["rejected_claims"])


def test_fake_joint_sensor_claim_is_rejected(evidence):
    modified=deepcopy(evidence)
    claim=next(c for c in modified["claims"] if c["claim_id"]=="sar_polarization")
    claim["sensor_view"]="JOINT"
    result=interpret_evidence(modified,"What does SAR show?")
    assert result["status"] == "UNAVAILABLE"
    rejected=next(c for c in result["rejected_claims"] if c["claim_id"]=="sar_polarization")
    assert any("sensor does not match" in reason for reason in rejected["reasons"])


def test_calibrated_confidence_injection_is_rejected(evidence):
    modified=deepcopy(evidence)
    claim=next(c for c in modified["claims"] if c["claim_id"]=="water")
    claim["confidence_status"]="99% confidence"
    result=interpret_evidence(modified,"What water evidence is present?")
    assert result["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("fabrication", ["17 houses detected", "A forest is definitely present", "A river is located at an exact address", "MODEL-PREDICTED vehicles"])
def test_fabricated_semantic_claim_text_is_rejected(evidence, fabrication):
    modified=deepcopy(evidence)
    claim=next(c for c in modified["claims"] if c["claim_id"]=="vegetation")
    claim["claim"]=fabrication
    result=interpret_evidence(modified,"What vegetation evidence is present?")
    assert result["status"] == "UNAVAILABLE"
    assert fabrication not in result["answer"]


def test_answer_is_exactly_deterministic(evidence):
    first=interpret_evidence(evidence,"Where is water located?",mode="technical")
    second=interpret_evidence(evidence,"Where is water located?",mode="technical")
    assert first == second


def test_every_emitted_claim_has_full_provenance(evidence):
    result=interpret_evidence(evidence,"What evidence supports the analysis?")
    for claim in result["claims"]:
        assert claim["source_features"] and claim["source_artifacts"]
        assert len(claim["token_indices"]) > 0
        assert claim["epistemic_status"] == "INFERRED_FROM_DERIVED_EVIDENCE"
    assert result["provenance"]["evidence_provenance"]


def test_invalid_mode_is_rejected(evidence):
    with pytest.raises(ValueError,match="simple or technical"):
        interpret_evidence(evidence,"What is present?",mode="creative")


def test_validator_detects_answer_tampering(evidence):
    result=interpret_evidence(evidence,"What water evidence is present?")
    result["answer"]="A fabricated answer."
    with pytest.raises(ValueError):
        validate_interpretation(result)
