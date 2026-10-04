"""Write the final v2 evidence-backed rejection receipt."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "semantic_improvement_v2"


def read(name: str): return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    runtime, splits = read("runtime_baseline.json"), read("splits.json")
    sar = read("experiments/experiment_01_sar_shuffle_ranking/result.json")
    joint = read("experiments/experiment_02_joint_shuffle_ranking/result.json")
    report = {
        "classification": "PROJECTOR_ONLY_CEILING_CONFIRMED",
        "deployment_recommendation": "DO_NOT_DEPLOY",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "starting_sha": "070e6eac1950ad710cb28c9cfd9a2f25abc1f5fb",
        "working_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "frozen_tag_target": subprocess.check_output(["git", "rev-parse", "satquery-sih-core-v1.0^{}"], text=True).strip(),
        "test_images_read": 0, "test_labels_read": 0, "test_inference_runs": 0, "test_metrics_computed": 0,
        "data_counts": {key: len(splits[key]) for key in ("train", "dev", "internal_holdout", "historically_exposed_locked_validation")},
        "dev_baseline": read("baseline_dev_metrics.json"),
        "sar_experiment": {"baseline": sar["baseline_dev_metrics"], "candidate": sar["candidate_dev_metrics"], "steps": sar["optimizer_steps"], "final_loss": sar["final_train_loss"]},
        "joint_experiment": {"baseline": joint["baseline_dev_metrics"], "candidate": joint["candidate_dev_metrics"], "steps": joint["optimizer_steps"], "final_loss": joint["final_train_loss"]},
        "historically_exposed_locked": {"baseline": read("baseline_locked_metrics.json"), "candidate": read("locked_validation_metrics.json")},
        "runtime": runtime,
        "conclusion": "Both projector-only shuffled-negative experiments reduced training loss but regressed DEV correctness. The selected joint candidate retained zero shuffled-SAR decision changes on the historical validation diagnostic. No fresh validation VQA identities remain in the approved manifest, so no acceptance-quality locked validation claim is possible.",
    }
    (OUT / "final_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("SATQUERY_SEMANTIC_IMPROVEMENT_V2_COMPLETE", flush=True)


if __name__ == "__main__": main()
