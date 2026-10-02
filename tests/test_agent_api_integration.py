from src.agent.api_integration import analyze_controller_request


def call(query, modality="rgb", inputs=1, temporal=None):
    return analyze_controller_request({"controller": True, "query": query,
        "inputs": [{"id": f"input-{n}", "modality": modality, "source_reference": "upload:fixture"} for n in range(inputs)], "temporal": temporal or {}})


def test_blocked_api_results_are_typed_and_secret_free():
    for query, modality, count, task, expected_status in (("What objects are in this SAR image?", "s1", 1, "SAR_VQA", "BLOCKED"),
                                                           ("Where is the river?", "rgb", 1, "GROUNDING", "BLOCKED"),
                                                           ("What changed between these two images?", "rgb", 2, "TEMPORAL_CHANGE_DESCRIPTION", "NOT_VERIFIED")):
        result = call(query, modality, count)
        assert result["status"] == expected_status and result["task_type"] == task
        reason = result["blocked_reason"] or result["final_answer"]["text"]
        assert "Fallback: NONE" in reason
        assert "C:\\" not in str(result)


def test_s2_is_not_silently_promoted_to_generation():
    result = call("Describe this Sentinel-2 image.", "s2")
    assert result["task_type"] == "S2_CAPTIONING"
    assert result["status"] == "NOT_VERIFIED"


def test_response_contains_provenance_and_complete_trace():
    result = call("Describe this image.")
    assert result["execution_id"] == result["provenance"]["execution_id"]
    assert [item["name"] for item in result["execution_trace"]["steps"]] == ["query_interpretation", "validation", "representation_selection", "tool_selection", "tool_execution", "evidence_collection", "reconciliation", "answer_generation"]
