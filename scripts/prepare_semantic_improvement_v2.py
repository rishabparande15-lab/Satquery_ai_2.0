"""Create the v2 audit and disjoint development split without touching TEST."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "semantic_improvement_v2"
MANIFEST = ROOT / "artifacts" / "training" / "phase3q" / "phase3q1_fusion_adaptation" / "phase3q1_training_manifest.json"
PRIOR = ROOT / "artifacts" / "semantic_improvement"
SEED = 20261004
TASKS = ("binary_qa", "multiple_choice_qa")


def dump(name: str, value: object) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact(row: dict) -> dict:
    return {key: row.get(key) for key in ("record_id", "image_id", "shuffled_image_id", "task_type", "question", "answer", "options", "split")}


def counts(rows: list[dict]) -> dict:
    return {
        "tasks": dict(Counter(row["task_type"] for row in rows)),
        "answers": dict(Counter(str(row["answer"]) for row in rows)),
        "mcq_positions": dict(Counter(str(row["answer"]) for row in rows if row["task_type"] == "multiple_choice_qa")),
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError("V2_NAMESPACE_EXISTS")
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if source.get("test_access_count") != 0:
        raise RuntimeError("SOURCE_MANIFEST_TEST_ACCESS_NOT_ZERO")
    prior_config = json.loads((PRIOR / "experiment_config.json").read_text(encoding="utf-8"))
    prior_summary = json.loads((PRIOR / "final_summary.json").read_text(encoding="utf-8"))
    prior_errors = json.loads((PRIOR / "error_analysis.json").read_text(encoding="utf-8"))
    prior_diagnostics = json.loads((PRIOR / "modality_diagnostic_summary.json").read_text(encoding="utf-8"))
    prior_ids = set(prior_config["validation_record_ids"])
    train = [dict(row) for row in source["selection"]["train"] if row["task_type"] in TASKS]
    validation = [dict(row) for row in source["selection"]["validation"] if row["task_type"] in TASKS]
    if len(train) != 40 or len(validation) != 20:
        raise RuntimeError("UNEXPECTED_VQA_POOL")
    if {row["record_id"] for row in validation} != prior_ids:
        raise RuntimeError("PRIOR_PANEL_NOT_EQUAL_TO_ALL_MANIFEST_VALIDATION_VQA")
    split = {"train": [], "dev": [], "internal_holdout": []}
    # Fixed 12/4/4 allocation per task.  The holdout is never used for model
    # selection; it is retained as an internal diagnostic only because the
    # base projectors were trained on the original manifest's train partition.
    for task in TASKS:
        rows = sorted((row for row in train if row["task_type"] == task), key=lambda row: row["record_id"])
        split["train"].extend(rows[:12])
        split["dev"].extend(rows[12:16])
        split["internal_holdout"].extend(rows[16:20])
    all_sets = {name: {row["image_id"] for row in rows} for name, rows in split.items()}
    overlaps = {f"{left}_{right}": sorted(all_sets[left] & all_sets[right]) for left, right in (("train", "dev"), ("train", "internal_holdout"), ("dev", "internal_holdout"))}
    if any(overlaps.values()):
        raise RuntimeError("V2_IMAGE_SPLIT_OVERLAP")
    audit = {
        "previous_improvement": "answer-format constrained candidate likelihood; it removed unparsed outputs and was not deployed",
        "prior_route_metrics": prior_summary["comparison"],
        "prior_answer_distributions": prior_errors,
        "prior_modality_diagnostics": prior_diagnostics,
        "visual_token_change_observed": True,
        "final_decision_invariance_observed": {"S2": "0/6", "SAR": "1/6", "joint_shuffled_sar": "0/6", "joint_full_controls": "0/18"},
        "answer_balance": prior_errors["panel_distribution"]["ground_truth_binary"],
        "mcq_position_balance": prior_errors["panel_distribution"]["ground_truth_mcq_option"],
        "supervision_audit": prior_errors["panel_distribution"]["supervision_construction_audit"],
        "locked_validation_status": "All 20 available validation VQA records were used by the prior experiment; no fresh validation VQA identities exist in this approved manifest.",
        "test_access_count": 0,
    }
    splits = {
        "seed": SEED,
        "selection_policy": "sorted record IDs; per-task 12 train, 4 dev, 4 internal holdout; original validation VQA is historically exposed final-comparison-only",
        "train": [compact(row) for row in split["train"]],
        "dev": [compact(row) for row in split["dev"]],
        "internal_holdout": [compact(row) for row in split["internal_holdout"]],
        "historically_exposed_locked_validation": [compact(row) for row in validation],
        "counts": {name: counts(rows) for name, rows in split.items()} | {"historically_exposed_locked_validation": counts(validation)},
        "image_overlap_checks": overlaps,
        "fresh_locked_validation_available": False,
        "test_access_count": 0,
    }
    dump("audit_previous_run.json", audit)
    dump("splits.json", splits)
    print("[V2] audit complete: prior panel consumed all validation VQA=20", flush=True)
    print("[V2] split complete: train=24 dev=8 internal_holdout=8 historically_exposed_locked=20 test=0", flush=True)


if __name__ == "__main__":
    main()
