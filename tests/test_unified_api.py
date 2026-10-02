from src.api import run_unified_query


def test_unified_api_returns_honest_blocked_grounding_without_loading_models():
    result = run_unified_query({
        "query": "Highlight the water body",
        "inputs": [{"role": "SINGLE", "modality": "s2", "input_id": "declared-safe-input"}],
    })
    assert result["status"] == "BLOCKED"
    assert result["route"] == "SINGLE_IMAGE_GROUNDING"
    assert result["error"]["code"] == "GROUNDING_MODEL_UNAVAILABLE"
    assert result["confidence"] == {"value": None, "type": "NOT_AVAILABLE"}


def test_unified_api_refuses_incomplete_temporal_input_before_execution():
    result = run_unified_query({
        "query": "What changed?",
        "inputs": [{"role": "T1", "modality": "optical", "input_id": "before"}],
    })
    assert result["status"] == "ERROR"
    assert result["error"]["code"] == "INCOMPLETE_TEMPORAL_PAIR"
