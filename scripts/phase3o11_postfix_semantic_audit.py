"""Phase 3O.11: exact Phase 3O.7 panel re-audit using the 3O.10 wrapper."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.phase3o6_semantic_audit import GENERATION_CONFIG, degenerate_flags, historical_s2_hash, normalize_binary, normalize_mcq, state_fingerprint
from scripts.phase3o7_task_balanced_audit import classification_summary, degeneracy_summary, paired
from scripts.run_phase3o2_adaptation_pilot import MODEL_REVISION, MODEL_SNAPSHOT, cube, hash_state
from src.eo_vlm.generation_interface import GENERATION_INTERFACE_VERSION, generate_with_multimodal_prefix
from src.eo_vlm.multispectral_projector import QWEN_IMAGE_GRID_THW, S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import S2_BANDS, inspect_area, sha256_file
from src.eo_vlm.training import freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

OUT = ROOT / "artifacts/training/phase3o"
RUN = OUT / "phase3o5_run"
CK = RUN / "s2_multispectral_projector.pt"
PANEL = OUT / "phase3o7_validation_panel.json"
SHUFFLE = OUT / "phase3o7_image_shuffle_map.json"
HIST = OUT / "phase3o7_task_balanced_semantic_audit.json"
ARTIFACT = OUT / "phase3o11_postfix_semantic_audit.json"
SHA = "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db"
FP = "db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
PANEL_SHA = "f59dc04057fbb1de2f917ff6a631643666eb7a0172cd3393c1a0eb060fd2a074"
SHUFFLE_SHA = "de10a2118511bd07bd26d893ccf929b864b70b5121cb279437e98fcb49920955"
TASKS = ("binary_qa", "multiple_choice_qa", "caption")
HISTORICAL = [RUN / "pilot_manifest.json", RUN / "training_summary.json", RUN / "phase3o5_reload_receipt.json", OUT / "phase3o6_semantic_audit.json", HIST, OUT / "phase3o8_visual_conditioning_diagnostic.json", OUT / "phase3o9_generation_supervision_diagnostic.json", OUT / "phase3o10_generation_interface_repair.json", PANEL, SHUFFLE]


def image_index():
    paths = {}
    for root in (Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"), Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2")):
        for path in root.rglob("*.tif"):
            if "_B" in path.stem:
                paths.setdefault(path.name, path)
    return paths


def termination(ids, eos):
    return "eos" if ids.numel() and int(ids[-1]) in set(eos) else "max_new_tokens"


def run_one(row, image_row, asset, runtime, model, projector, condition):
    question = row["question"] if row["task_type"] != "caption" else "Describe this Sentinel-2 image."
    reference = row["answer"] if row["task_type"] != "caption" else row["caption"]
    ids = [model.config.vision_start_token_id] + [model.config.image_token_id] * 16 + [model.config.vision_end_token_id] + runtime["processor"].tokenizer(f"Question: {question}\nAnswer:", add_special_tokens=True)["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    attention = torch.ones_like(input_ids)
    embeds = model.get_input_embeddings()(input_ids)
    visual = projector(cube(asset).to("cuda"))
    mask = input_ids.eq(model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
    embeds = embeds.masked_scatter(mask, visual.to(embeds.dtype).reshape(-1))
    generated = generate_with_multimodal_prefix(model, input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention,
        image_grid_thw=torch.tensor([QWEN_IMAGE_GRID_THW], dtype=torch.long, device="cuda"), max_new_tokens=16,
        do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, use_cache=True)[0, input_ids.shape[1]:]
    text = runtime["processor"].tokenizer.decode(generated, skip_special_tokens=True)
    if row["task_type"] == "binary_qa": prediction, parse = normalize_binary(text)
    elif row["task_type"] == "multiple_choice_qa": prediction, parse = normalize_mcq(text, row.get("options"))
    else: prediction, parse = (text.strip() or None), "caption_descriptive_only"
    image_paths = {band: Path(asset["local_path"]) / f"{image_row['image_id']}_{band}.tif" for band in S2_BANDS}
    return {"condition": condition, "sample_id": row["record_id"], "task_type": row["task_type"], "question": question,
        "reference": reference, "allowed_answer_choices": row.get("options"), "image_id_used": image_row["image_id"],
        "image_sample_id_used": image_row["record_id"], "image_s2_source_hash": historical_s2_hash(image_paths),
        "generated_text": text, "generated_token_ids": [int(x) for x in generated], "generation_token_count": int(generated.numel()),
        "termination_reason": termination(generated, model.generation_config.eos_token_id), "normalized_prediction": prediction,
        "parse_status": parse, "correct": prediction == reference if row["task_type"] != "caption" and prediction is not None else None,
        "degenerate_flags": degenerate_flags(text, f"Question: {question}\nAnswer:"), "generation_interface_version": GENERATION_INTERFACE_VERSION}


def hash_files(paths): return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def caption_summary(rows):
    base = degeneracy_summary(rows)
    base["nonempty"] = len(rows) - base["empty"]
    base["termination_reasons"] = dict(Counter(x["termination_reason"] for x in rows))
    return base


def main():
    started = time.perf_counter()
    if sha256_file(CK) != SHA or state_fingerprint(torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]) != FP: raise RuntimeError("PHASE3O11_BLOCKED_MODEL_IDENTITY")
    if sha256_file(PANEL) != PANEL_SHA or sha256_file(SHUFFLE) != SHUFFLE_SHA: raise RuntimeError("PHASE3O11_BLOCKED_HISTORICAL_INPUTS")
    panel, shuffle = json.loads(PANEL.read_text()), json.loads(SHUFFLE.read_text())
    if {task: len(panel["records"].get(task, [])) for task in TASKS} != {task: 30 for task in TASKS} or len(shuffle["mapping"]) != 90: raise RuntimeError("PHASE3O11_BLOCKED_PANEL_CONTENT")
    rows = {r["record_id"]: r for task in TASKS for r in panel["records"][task]}
    mapping = {m["sample_id"]: m for m in shuffle["mapping"]}
    if set(mapping) != set(rows): raise RuntimeError("PHASE3O11_BLOCKED_MAPPING_CONTENT")
    historical_before = hash_files(HISTORICAL)
    ix = image_index(); assets = {rid: inspect_area(row, {b: ix[f"{row['image_id']}_{b}.tif"] for b in S2_BANDS}).to_dict() for rid, row in rows.items()}
    runtime = Qwen25VLRGBAdapter().load_model(MODEL_SNAPSHOT, dtype="float16", device="cuda")
    model = runtime["model"].eval()
    if freeze_qwen(model)["qwen_trainable_parameter_count"]: raise RuntimeError("PHASE3O11_BLOCKED_QWEN_TRAINABLE")
    projector = S2MultispectralProjector().to("cuda"); projector.load_state_dict(torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]); projector.eval()
    q_before, p_before = hash_state(model), hash_state(projector)
    normal, shuffled = [], []
    with torch.inference_mode():
        for task in TASKS:
            for row in panel["records"][task]:
                normal.append(run_one(row, row, assets[row["record_id"]], runtime, model, projector, "normal"))
                target = rows[mapping[row["record_id"]]["shuffled_sample_id"]]
                shuffled.append(run_one(row, target, assets[target["record_id"]], runtime, model, projector, "shuffled_image"))
    q_after, p_after, historical_after = hash_state(model), hash_state(projector), hash_files(HISTORICAL)
    if q_before != q_after or p_before != p_after or historical_before != historical_after or sha256_file(CK) != SHA: raise RuntimeError("PHASE3O11_FAILED_IMMUTABILITY")
    by_task = lambda values: {t: [x for x in values if x["task_type"] == t] for t in TASKS}
    n, s = by_task(normal), by_task(shuffled)
    old = json.loads(HIST.read_text())
    change = {t: sum(a["generated_text"] != b["generated_text"] for a, b in zip(old["normal_results"], normal) if a["task_type"] == t) for t in TASKS}
    caps_changed = sum(a["generated_text"] != {x["sample_id"]: x for x in s["caption"]}[a["sample_id"]]["generated_text"] for a in n["caption"])
    bn, bs, mn, ms = classification_summary(n["binary_qa"]), classification_summary(s["binary_qa"]), classification_summary(n["multiple_choice_qa"]), classification_summary(s["multiple_choice_qa"])
    delta_b, delta_m = bn["accuracy"] - bs["accuracy"], mn["accuracy"] - ms["accuracy"]
    classification = "STRONGER_IMAGE_DEPENDENCE_OBSERVED" if delta_b >= 0.1 and delta_m >= 0.1 else "WEAK_IMAGE_DEPENDENCE" if abs(delta_b) > 0 or abs(delta_m) > 0 else "NO_IMAGE_DEPENDENCE_ESTABLISHED"
    artifact = {"phase": "3O.11", "status": "PHASE3O11_COMPLETE", "audit_scope": "exact_phase3o7_postfix_rerun", "generation_interface_version": GENERATION_INTERFACE_VERSION,
        "test_access_count": 0, "panel": {"path": str(PANEL.relative_to(ROOT)), "sha256": PANEL_SHA, "reused_exactly": True, "counts": {t: len(panel["records"][t]) for t in TASKS}},
        "shuffle_mapping": {"path": str(SHUFFLE.relative_to(ROOT)), "sha256": SHUFFLE_SHA, "reused_exactly": True, "count": len(mapping)},
        "normal_results": normal, "shuffled_results": shuffled, "normal_task_results": {"binary_qa": bn, "multiple_choice_qa": mn}, "shuffled_task_results": {"binary_qa": bs, "multiple_choice_qa": ms},
        "accuracy_deltas_normal_minus_shuffled": {"binary_qa": delta_b, "multiple_choice_qa": delta_m}, "paired_correctness": {"binary_qa": paired(n["binary_qa"], s["binary_qa"]), "multiple_choice_qa": paired(n["multiple_choice_qa"], s["multiple_choice_qa"])},
        "caption_image_dependence": {"normal": caption_summary(n["caption"]), "shuffled": caption_summary(s["caption"]), "identical_generation_count": 30 - caps_changed, "changed_generation_count": caps_changed},
        "normal_degeneracy": degeneracy_summary(normal), "shuffled_degeneracy": degeneracy_summary(shuffled), "prefix_vs_postfix": {"phase3o7": old["normal_task_results"], "phase3o11": {"normal": {"binary_qa": bn, "multiple_choice_qa": mn}, "shuffled": {"binary_qa": bs, "multiple_choice_qa": ms}}, "changed_outputs_by_task": change},
        "image_dependence_classification": classification, "checkpoint_sha256": SHA, "checkpoint_sha256_verified": True, "projector_fingerprint": FP, "projector_fingerprint_verified": True, "qwen_revision": MODEL_REVISION, "qwen_immutability_verified": True, "projector_immutability_verified": True, "historical_artifacts_unchanged": True, "no_optimizer_created": True, "no_backward_pass": True, "duration_seconds": time.perf_counter() - started}
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": artifact["status"], "normal": artifact["normal_task_results"], "shuffled": artifact["shuffled_task_results"], "caption": artifact["caption_image_dependence"], "classification": classification}, indent=2))

if __name__ == "__main__": main()
