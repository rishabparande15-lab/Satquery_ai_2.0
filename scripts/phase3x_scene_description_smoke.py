"""Run the official Phase 3X scene-description specialist on one approved train image.

This is a functional smoke check, not a benchmark evaluation.  It uses no test
records and writes only a compact receipt, never image pixels.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.dataset_loader import discover_s2_samples, load_optical_sample


DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", r"D:\Satquery_ai datasets\comparison\raw-1000"))
MODEL_ROOT = Path(r"D:\Satquery_ai datasets\checkpoints\remote-sensing-Qwen2-VL-2B-Instruct")
PATCH_ID = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
OUT = PROJECT / "artifacts" / "final" / "sih" / "phase3x"


def rgb_from_s2(optical: np.ndarray) -> Image.Image:
    """Build an explicit B04/B03/B02 display rendition; not sensor inference."""
    rgb = np.stack((optical[3], optical[2], optical[1]), axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Cannot render finite contrast-stretched RGB input")
    scaled = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return Image.fromarray((scaled * 255).round().astype(np.uint8), mode="RGB")


def main() -> None:
    samples = {sample.patch_id: sample for sample in discover_s2_samples(DATASET_ROOT, allowed_splits=("train",))}
    sample = samples.get(PATCH_ID)
    if sample is None:
        raise FileNotFoundError(f"Approved train patch is absent: {PATCH_ID}")
    prepared = load_optical_sample(sample)
    image = rgb_from_s2(prepared.raw_optical)
    started = perf_counter()
    # The official checkpoint was saved with Transformers 4.45's min/max-pixel
    # processor schema.  Construct the documented processor directly so 4.51
    # does not reinterpret its legacy ``size`` mapping.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    image_processor = Qwen2VLImageProcessor(min_pixels=3136, max_pixels=12845056)
    processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.chat_template = json.loads((MODEL_ROOT / "chat_template.json").read_text(encoding="utf-8"))["chat_template"]
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ROOT, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True,
    ).eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this remote-sensing image in one concise sentence."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    description = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    if not description:
        raise RuntimeError("Scene-description specialist returned empty text")
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "COMPLETED", "task": "SINGLE_IMAGE_SCENE_DESCRIPTION",
        "specialist": "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct",
        "model_sha256": hashlib.sha256((MODEL_ROOT / "model.safetensors").read_bytes()).hexdigest(),
        "model_revision": "main", "license": "Apache-2.0", "device": "cuda:0",
        "input": {"patch_id": PATCH_ID, "split": "train", "source": "approved_local_bigearthnet", "rendering": "B04/B03/B02 contrast-stretched RGB"},
        "description": description, "runtime_seconds": round(perf_counter() - started, 3),
        "test_access": 0,
    }
    (OUT / "phase3x_second_single_image_result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
