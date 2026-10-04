"""Complete safe joint shuffle controls on the predeclared validation subset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_semantic_improvement import (OUT, close, constrained, dump,
    load_panel, make_controller, validation_samples, visual)


def paired(base, *, optical, sar):
    return type(base)(base.patch_id, optical, sar, base.raw_optical, base.raw_sar,
                      base.reference, base.metadata)


def main() -> None:
    rows = load_panel()
    subset = [row for task in ("binary_qa", "multiple_choice_qa") for row in [item for item in rows if item["task_type"] == task][:3]]
    samples = validation_samples(subset)
    controller = make_controller("JOINT")
    controller.load()
    records = []
    try:
        for index, row in enumerate(subset, 1):
            correct = samples[row["image_id"]]
            shuffled = samples[row["shuffled_image_id"]]
            conditions = {
                "correct_s2_correct_sar": correct,
                "correct_s2_shuffled_sar": paired(correct, optical=correct.optical, sar=shuffled.sar),
                "shuffled_s2_correct_sar": paired(correct, optical=shuffled.optical, sar=correct.sar),
            }
            reference = constrained(controller, "JOINT", row, correct)
            reference_tokens = visual(controller, "JOINT", correct)
            for condition, prepared in conditions.items():
                result = constrained(controller, "JOINT", row, prepared)
                tokens = visual(controller, "JOINT", prepared)
                records.append({
                    "record_id": row["record_id"], "task_type": row["task_type"],
                    "condition": condition, "prediction": result["prediction"],
                    "reference_prediction": reference["prediction"],
                    "decision_changed": result["prediction"] != reference["prediction"],
                    "visual_token_l2_vs_correct": float(torch.linalg.vector_norm(reference_tokens - tokens).detach().cpu()),
                })
            print(f"[JOINT-DIAGNOSTIC {index}/6] complete", flush=True)
    finally:
        close(controller)
    old = json.loads((OUT / "modality_diagnostics.json").read_text(encoding="utf-8"))
    old["JOINT_FULL_CONTROLS"] = records
    dump("modality_diagnostics.json", old)
    dump("joint_modality_controls.json", records)
    print("SATQUERY_JOINT_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
