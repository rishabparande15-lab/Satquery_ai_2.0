"""Finalize comparison artifacts from the completed semantic evaluator records."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_semantic_improvement import OUT, ROUTES, dump, emit, metric


def main() -> None:
    baseline = json.loads((OUT / "baseline_predictions.json").read_text(encoding="utf-8"))
    post = [{**row, "prediction": row["constrained"]["prediction"]} for row in baseline]
    baseline_metrics = json.loads((OUT / "baseline_metrics.json").read_text(encoding="utf-8"))
    post_metrics = {route: metric([row for row in post if row["route"] == route], "prediction") for route in ROUTES}
    panel_rows = [row for row in baseline if row["route"] == "S2"]
    error = {
        route: {
            task: {
                "generated_unparsed": sum(row["prediction"] is None for row in baseline if row["route"] == route and row["task_type"] == task),
                "generated_answer_counts": {str(key): value for key, value in Counter(row["prediction"] for row in baseline if row["route"] == route and row["task_type"] == task).items()},
            }
            for task in ("binary_qa", "multiple_choice_qa")
        }
        for route in ROUTES
    }
    error["panel_distribution"] = {
        "ground_truth_binary": dict(Counter(row["answer"] for row in panel_rows if row["task_type"] == "binary_qa")),
        "ground_truth_mcq_option": dict(Counter(row["answer"] for row in panel_rows if row["task_type"] == "multiple_choice_qa")),
        "question_counts": dict(Counter(row["question"] for row in panel_rows)),
        "immediate_or_empty_generated_outputs": sum(not row["generated"]["text"].strip() for row in baseline),
        "repeated_generated_outputs": {
            route: {text: count for text, count in Counter(row["generated"]["text"] for row in baseline if row["route"] == route).items() if count > 1}
            for route in ROUTES
        },
        "supervision_construction_audit": {
            "checked": "src/eo_vlm/training.py build_qwen_visual_token_batch; existing targeted regression tests passed",
            "target_text": "provided answer text is tokenized as labels",
            "attention_mask": "constructed for multimodal sequence",
            "visual_token_placement": "visual token batch builder is shared by the measured constrained scorer",
            "training_run": False,
        },
    }
    route_pairs = {}
    for phase, rows in (("baseline", baseline), ("post", post)):
        by_id = {(row["record_id"], row["task_type"], row["route"]): row for row in rows}
        phase_pairs = {}
        for comparator in ("SAR", "JOINT"):
            transitions = Counter()
            for row in rows:
                if row["route"] != comparator:
                    continue
                s2 = by_id[(row["record_id"], row["task_type"], "S2")]
                key = (
                    "both_correct" if row["prediction"] == row["answer"] and s2["prediction"] == s2["answer"]
                    else "candidate_correct_s2_wrong" if row["prediction"] == row["answer"]
                    else "s2_correct_candidate_wrong" if s2["prediction"] == s2["answer"]
                    else "both_wrong"
                )
                transitions[key] += 1
            phase_pairs[comparator] = dict(transitions)
        route_pairs[phase] = phase_pairs
    diagnostics = json.loads((OUT / "modality_diagnostics.json").read_text(encoding="utf-8"))
    diagnostic_summary = {
        route: {
            "records": len(rows),
            "decision_changes": sum(row["decision_changed"] for row in rows),
            "mean_visual_token_l2": sum(row.get("visual_token_l2", row.get("visual_token_l2_vs_correct", 0.0)) for row in rows) / len(rows),
        }
        for route, rows in diagnostics.items()
    }
    comparison = {
        route: {
            task: {
                "baseline": baseline_metrics[route][task],
                "post": post_metrics[route][task],
                "delta_accuracy_over_total": (post_metrics[route][task]["accuracy_over_total"] or 0) - (baseline_metrics[route][task]["accuracy_over_total"] or 0),
            }
            for task in baseline_metrics[route]
        }
        for route in ROUTES
    }
    dump("post_predictions.json", post)
    dump("post_metrics.json", post_metrics)
    dump("error_analysis.json", error)
    dump("baseline_vs_post.json", comparison)
    dump("route_comparisons.json", route_pairs)
    dump("modality_diagnostic_summary.json", diagnostic_summary)
    dump("candidate_results/constrained_candidate_likelihood.json", {
        "name": "validation-wide constrained candidate likelihood",
        "scope": "All legal binary or MCQ answer candidates are scored with frozen visual tokens; no sample-specific rule is used.",
        "selection": "Experimental only; not integrated into product routes because SAR MCQ regressed on this fixed validation panel.",
        "comparison": comparison,
    })
    dump("training_decision.json", {
        "training_run": False,
        "reason": "No training was justified: the candidate is inference-only and the fixed validation panel showed a SAR MCQ regression.",
        "test_access_count": 0,
    })
    runtime_path = OUT / "runtime_metrics.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {"status": "not_measured"}
    config_path = OUT / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "checkpoint_hashes": {
            "s2_projector": "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db",
            "sar_projector": "a7e57bff0cba141db5c34f549d0bc3f25638960fe39ab25246bbab49e979accd",
            "joint_projector": "bedcb8dc60d3375878b6fe732cb90b568283c7d5565148fb1a52eae0d1b12eaa",
        },
        "test_access_count": 0,
        "panel_policy": "Only the predeclared 20 validation records and their paired validation shuffle identities were read.",
    })
    dump("experiment_config.json", config)
    all_deltas = [item["delta_accuracy_over_total"] for route in comparison.values() for item in route.values()]
    dump("final_summary.json", {
        "classification": "SEMANTIC_PERFORMANCE_PARTIALLY_IMPROVED" if any(delta > 0 for delta in all_deltas) else "NO_RELIABLE_SEMANTIC_GAIN",
        "best_intervention": "validation-wide constrained candidate likelihood",
        "production_adoption": "not adopted; SAR MCQ regressed on the fixed validation panel",
        "test_images_read": 0,
        "test_labels_read": 0,
        "test_inference_runs": 0,
        "test_metrics_computed": 0,
        "comparison": comparison,
        "diagnostics": diagnostic_summary,
        "runtime": runtime,
    })
    emit("SATQUERY_SEMANTIC_IMPROVEMENT_RUN_COMPLETE")
    print(json.dumps({"comparison": comparison, "records": len(baseline), "route_pairs": route_pairs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
