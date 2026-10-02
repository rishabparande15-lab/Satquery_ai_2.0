"""One-record, validation-only Phase 3AD SAR route verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.dataset_loader import discover_samples, load_sample
from src.single_image_sar_vqa import SingleImageSARVQAController
from src.eo_vlm.training import build_qwen_visual_token_batch
from src.eo_vlm.image_conditioned_objective import target_sequence_log_probability


MANIFEST = ROOT / "artifacts/training/phase3p/phase3p1_sar_adaptation/phase3p1_training_manifest.json"


def main() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))["selection"]["validation"][0]
    sample_id = record["image_id"]
    print("PHASE3AD validation-only record selected", flush=True)
    sample = next(item for item in discover_samples(get_settings().dataset_root, strict=True) if item.patch_id == sample_id)
    prepared = load_sample(sample)
    sar = prepared.sar.astype("float32", copy=False)
    digest = hashlib.sha256(sar.tobytes()).hexdigest()
    torch.cuda.reset_peak_memory_stats()
    controller = SingleImageSARVQAController()
    try:
        result = controller.run(image_id=sample_id, split="validation", sar=sar,
            metadata={**prepared.metadata, "sar_band_order": ["VV", "VH"]},
            question=record["question"], task_type=record["task_type"])
        shuffled_id = record["shuffled_image_id"]
        shuffled = load_sample(next(item for item in discover_samples(get_settings().dataset_root, strict=True) if item.patch_id == shuffled_id)).sar.astype("float32", copy=False)
        assert controller.croma is not None and controller.projector is not None and controller.model is not None and controller.tokenizer is not None
        # CROMA's inference-mode tensors must be copied into ordinary tensors
        # before the projector's autograd-aware modules consume them.
        correct_tokens = torch.from_numpy(controller.croma.infer_modality(sar=sar)["SAR_encodings"].detach().float().cpu().numpy().copy()).to(controller.device)
        shuffled_tokens = torch.from_numpy(controller.croma.infer_modality(sar=shuffled)["SAR_encodings"].detach().float().cpu().numpy().copy()).to(controller.device)
        with torch.no_grad():
            correct_visual, shuffled_visual = controller.projector(correct_tokens), controller.projector(shuffled_tokens)
        def scores(visual):
            values = {}
            for candidate in ("yes", "no"):
                batch = build_qwen_visual_token_batch(tokenizer=controller.tokenizer, model=controller.model, visual_tokens=visual, questions=[record["question"]], answers=[candidate])
                values[candidate] = float(target_sequence_log_probability(controller.model(**batch).logits, batch["labels"]))
            return values
        correct_scores, shuffled_scores = scores(correct_visual), scores(shuffled_visual)
        conditioning = {"shuffled_sample_id": shuffled_id, "projected_token_l2": float(torch.linalg.vector_norm(correct_visual - shuffled_visual)),
            "correct_candidate_scores": correct_scores, "shuffled_candidate_scores": shuffled_scores,
            "correct_decision": max(correct_scores, key=correct_scores.get), "shuffled_decision": max(shuffled_scores, key=shuffled_scores.get)}
        receipt = {"sample_id": sample_id, "split": "validation", "input_sha256": digest,
            "question": record["question"], "expected_answer": record.get("answer"),
            "answer": result["answer"], "parsed_answer": result["parsed_answer"], "route": result["route"],
            "visual_evidence": result["visual_evidence"], "warnings": result["warnings"],
            "provenance": result["provenance"], "max_cuda_allocated": torch.cuda.max_memory_allocated(),
            "max_cuda_reserved": torch.cuda.max_memory_reserved(), "conditioning": conditioning, "test_access": 0}
        print("PHASE3AD_DIRECT_RESULT=" + json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        controller.close()
        torch.cuda.empty_cache()


if __name__ == "__main__": main()
