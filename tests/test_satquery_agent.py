from src.satquery_agent import (SatQueryAgent, capability_registry, classify_input_configuration,
                                classify_query_intent, select_route)


def _agent(calls):
    def execute(payload):
        calls.append(payload["route"])
        if payload["route"] == "TEMPORAL_CHANGE_DESCRIPTION":
            return {"change_description": "a building was added", "t1_identity": {"id": "pre"}, "t2_identity": {"id": "post"}, "provenance": {"model": "Chg2Cap"}}
        return {"answer": "yes", "selected_specialist": payload["route"], "provenance": {"verified": True}, "visual_evidence": {"type": "input"}}
    return SatQueryAgent({"SINGLE_IMAGE_VQA": execute, "OPTICAL_SAR_ANALYSIS": execute, "TEMPORAL_CHANGE_DESCRIPTION": execute})


def test_registry_discloses_maturity_without_overclaiming():
    registry = capability_registry()
    assert registry["TEMPORAL_CHANGE_DESCRIPTION"]["status"] == "AVAILABLE"
    assert registry["SINGLE_IMAGE_GROUNDING"]["status"] == "BLOCKED"
    assert registry["SINGLE_IMAGE_CAPTION"]["status"] == "EXPERIMENTAL_LIMITED"


def test_configuration_and_intent_drive_each_available_specialist():
    calls = []; agent = _agent(calls)
    cases = [
        ([{"role": "SINGLE", "modality": "s2"}], "Is water visible?", "SINGLE_IMAGE_VQA"),
        ([{"role": "S1", "modality": "s1"}, {"role": "S2", "modality": "s2"}], "Use SAR and optical information together.", "OPTICAL_SAR_ANALYSIS"),
        ([{"role": "T1", "modality": "optical"}, {"role": "T2", "modality": "optical"}], "What changed between these two images?", "TEMPORAL_CHANGE_DESCRIPTION"),
    ]
    for inputs, query, route in cases:
        result = agent.run(inputs=inputs, query=query)
        assert result["status"] == "COMPLETED"
        assert result["route"] == route
        assert result["confidence"] == {"value": None, "type": "NOT_AVAILABLE"}
        assert result["execution_trace"]
    assert calls == [case[2] for case in cases]


def test_grounding_and_scene_description_do_not_silently_fallback():
    agent = _agent([]); inputs = [{"role": "SINGLE", "modality": "s2"}]
    grounding = agent.run(inputs=inputs, query="Highlight the water body")
    scene = agent.run(inputs=inputs, query="Describe this image")
    assert grounding["status"] == "BLOCKED" and grounding["route"] == "SINGLE_IMAGE_GROUNDING"
    assert grounding["error"]["code"] == "GROUNDING_MODEL_UNAVAILABLE"
    assert scene["status"] == "ERROR" and scene["route"] == "SINGLE_IMAGE_SCENE_DESCRIPTION"
    assert scene["error"]["code"] == "SPECIALIST_UNAVAILABLE"


def test_incomplete_and_ambiguous_configurations_fail_closed():
    agent = _agent([])
    incomplete = agent.run(inputs=[{"role": "T1", "modality": "optical"}], query="What changed?")
    mismatch = agent.run(inputs=[{"role": "T1", "modality": "optical"}, {"role": "T2", "modality": "optical"}], query="What land cover is visible?")
    assert incomplete["error"]["code"] == "INCOMPLETE_TEMPORAL_PAIR"
    assert mismatch["error"]["code"] == "ROUTING_CLARIFICATION_REQUIRED"


def test_override_is_validated_against_input_and_intent():
    calls = []; agent = _agent(calls)
    good = agent.run(inputs=[{"role": "SINGLE", "modality": "s2"}], query="Is water visible?", requested_task="SINGLE_IMAGE_VQA")
    bad = agent.run(inputs=[{"role": "SINGLE", "modality": "s2"}], query="Is water visible?", requested_task="OPTICAL_SAR_ANALYSIS")
    assert good["route"] == "SINGLE_IMAGE_VQA"
    assert bad["error"]["code"] == "REQUESTED_TASK_INCOMPATIBLE"


def test_classifiers_are_deterministic():
    configuration = classify_input_configuration([{"role": "S1", "modality": "s1"}, {"role": "S2", "modality": "s2"}])
    intent, _ = classify_query_intent("Use SAR and optical information together")
    assert configuration.kind == "CROSS_MODAL_PAIR"
    assert select_route(configuration, intent)[0] == "OPTICAL_SAR_ANALYSIS"
