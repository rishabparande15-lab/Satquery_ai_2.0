import numpy as np
import pytest

from src.eo_vlm.optical_sar_joint import (JointInputError, load_verified_joint_projector,
    parse_answer, response_failure, run_optical_sar_joint, validate_paired_inputs)


def valid_kwargs():
    return {"s1": np.zeros((2, 120, 120), dtype=np.float32), "s2": np.zeros((12, 120, 120), dtype=np.float32),
            "s1_patch_id": "patch-1", "s2_patch_id": "patch-1", "expected_patch_id": "patch-1",
            "metadata": {"crs": "EPSG:32631", "resolution": [10.0, 10.0], "bounds": [0.0, 0.0, 1200.0, 1200.0]}}


def test_paired_contract_accepts_canonical_pair():
    validate_paired_inputs(**valid_kwargs())


@pytest.mark.parametrize("field,reason", [("s1", "MISSING_S1"), ("s2", "MISSING_S2")])
def test_missing_modalities_fail_closed(field, reason):
    values = valid_kwargs(); values[field] = None
    with pytest.raises(JointInputError, match=reason): validate_paired_inputs(**values)


@pytest.mark.parametrize("field,shape", [("s1", (1, 120, 120)), ("s2", (11, 120, 120))])
def test_bad_shapes_fail_closed(field, shape):
    values = valid_kwargs(); values[field] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(JointInputError, match="INVALID"): validate_paired_inputs(**values)


def test_nonfinite_and_identity_mismatch_fail_closed():
    values = valid_kwargs(); values["s1"][0, 0, 0] = np.nan
    with pytest.raises(JointInputError, match="NONFINITE"): validate_paired_inputs(**values)
    values = valid_kwargs(); values["s2_patch_id"] = "other"
    with pytest.raises(JointInputError, match="MISMATCHED_PATCH_ID"): validate_paired_inputs(**values)


def test_output_contract_helpers():
    failure = response_failure(record_id="r", task_type="caption", reason="MISSING_S1")
    assert set(failure) == {"record_id", "task_type", "generated_text", "parsed_answer", "error_status", "provenance", "diagnostics"}
    assert parse_answer("binary_qa", "Yes.") == "yes"
    assert parse_answer("multiple_choice_qa", "(c)") == "c"


def test_missing_checkpoint_is_rejected(tmp_path):
    with pytest.raises(JointInputError, match="MISSING_JOINT_CHECKPOINT"):
        load_verified_joint_projector(tmp_path / "missing.pt", device="cpu")


def test_missing_croma_runtime_is_structured_failure():
    class MissingCroma:
        def infer(self, *_args): raise FileNotFoundError("missing")
    values = valid_kwargs()
    response = run_optical_sar_joint(croma=MissingCroma(), projector=None, model=None, tokenizer=None,
        record_id="r", task_type="caption", question="describe", qwen_revision="pinned",
        joint_checkpoint_sha="x", max_new_tokens=1, **values)
    assert response["error_status"] == "MISSING_CROMA_ASSET"
