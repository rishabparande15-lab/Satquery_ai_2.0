from src.eo_vlm.phase3q3_metrics import CONDITIONS, changed, evidence_gate, outcome, paired_wins, qa_summary, task_classification


def row(parsed, target="yes", text="x"): return {"parsed_answer": parsed, "target": target, "generated_text": text}


def test_condition_set_and_qa_comparison_are_deterministic():
    assert len(CONDITIONS) == 7
    s2, joint = [row("yes"), row(None)], [row("yes"), row("yes")]
    assert qa_summary(s2)["accuracy"] == 0.5
    assert paired_wins(s2, joint) == {"joint_wins": 1, "joint_losses": 0, "ties": 1}
    assert task_classification(qa_summary(s2), qa_summary(joint)) == "JOINT_BETTER"


def test_changes_and_route_gate():
    assert changed([row("yes", text="a")], [row("yes", text="b")]) == 1
    assert evidence_gate(task_classes={"binary": "NO_CLEAR_DIFFERENCE", "mcq": "NO_CLEAR_DIFFERENCE"}, s1_changes=0, s2_changes=3, total_records=20)[0] == "OPTICAL_SAR_TECHNICALLY_VALID_S2_DOMINANT"
    assert outcome(row(None)) == "unparsable"
