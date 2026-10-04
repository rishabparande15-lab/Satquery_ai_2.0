"""Prepare the final LoRA-only experiment split and audit receipts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "artifacts" / "semantic_improvement_v2"
OUT = ROOT / "artifacts" / "semantic_improvement_v3"


def dump(name: str, value: object) -> None:
    target = OUT / name; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def counts(rows: list[dict]) -> dict:
    return {"tasks": dict(Counter(row["task_type"] for row in rows)), "answers": dict(Counter(str(row["answer"]) for row in rows)), "mcq_positions": dict(Counter(str(row["answer"]) for row in rows if row["task_type"] == "multiple_choice_qa"))}


def main() -> None:
    if OUT.exists(): raise RuntimeError("V3_NAMESPACE_EXISTS")
    split = json.loads((V2 / "splits.json").read_text(encoding="utf-8"))
    final = json.loads((V2 / "final_summary.json").read_text(encoding="utf-8"))
    audit = {"v2_classification": final["classification"], "v2_conclusion": final["conclusion"], "v2_metrics": {"sar": final["sar_experiment"], "joint": final["joint_experiment"]}, "decoding": "legal answer candidate likelihood over yes/no or MCQ option keys", "projector_hashes": {"s2": "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db", "sar": "a7e57bff0cba141db5c34f549d0bc3f25638960fe39ab25246bbab49e979accd", "joint": "bedcb8dc60d3375878b6fe732cb90b568283c7d5565148fb1a52eae0d1b12eaa"}, "peft_available": False, "observed_qwen_attention_targets": ["model.layers.35.self_attn.q_proj", "model.layers.35.self_attn.v_proj"], "test_access_count": 0}
    selected = {"train": split["train"], "dev": split["dev"], "locked_validation": split["internal_holdout"]}
    identities = {name: {row["image_id"] for row in rows} for name, rows in selected.items()}
    overlaps = {f"{a}_{b}": sorted(identities[a] & identities[b]) for a, b in (("train", "dev"), ("train", "locked_validation"), ("dev", "locked_validation"))}
    if any(overlaps.values()): raise RuntimeError("V3_SPLIT_OVERLAP")
    receipts = {"seed": 20261004, "selection": "V2 train/dev reused; V2 internal_holdout becomes final LoRA locked panel and was not used for V2 fitting or selection.", "limitation": "The holdout derives from original source-train data and may have informed the pre-existing projector training; it is nevertheless untouched by V2/V3 candidate fitting and selection.", **selected, "counts": {name: counts(rows) for name, rows in selected.items()}, "image_overlap_checks": overlaps, "test_access_count": 0}
    dump("audit.json", audit); dump("splits.json", receipts)
    print("[V3] audit and split complete: train=24 dev=8 locked=8 test=0", flush=True)


if __name__ == "__main__": main()
