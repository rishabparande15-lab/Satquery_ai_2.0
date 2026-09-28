"""Run the bounded Phase 3G learned S2 projector pilot.

The script consumes only real, split-validated BigEarthNet.txt visual-VQA
records with all twelve S2 TIFFs available.  It never reads test records and
never touches scientific representations or checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import psutil
import rasterio
import torch
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_loader import OPTICAL_BANDS, _read_to_grid
from src.eo_vlm_adapter import Qwen25VLRGBAdapter
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.training import (
    PilotConfig,
    build_qwen_token_batch,
    freeze_qwen,
    save_projector_checkpoint,
)
from src.preprocessing import _robust_channel_scale


DEFAULT_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET_FINGERPRINT = "7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625"
SPLIT_FINGERPRINT = "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"
ANNOTATION_FINGERPRINT = "5562e7976aee140850fede7efe5d1c375934244fe5c0c4a754b76eeb1af5447b"
ANNOTATION_SOURCE_REVISION = "72d865f2146f0a85b720f7f3ca1cdbaeafc3d316"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _area_source_hash(paths: dict[str, Path]) -> str:
    payload = [(band, _sha256_file(paths[band])) for band in OPTICAL_BANDS]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _load_records(path: Path, names: set[str], *, split: str, limit: int) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("split") != split or record.get("pool") != "visual_vqa_candidate":
                continue
            area = record.get("image_id")
            if not isinstance(area, str) or not all(f"{area}_{band}.tif" in names for band in OPTICAL_BANDS):
                continue
            records.append(record)
            if len(records) == limit:
                break
    if len(records) != limit:
        raise RuntimeError(f"only found {len(records)} real {split} records; required {limit}")
    return records


def _build_path_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.tif"):
            if path.name in index:
                raise RuntimeError(f"duplicate raw S2 filename across roots: {path.name}")
            index[path.name] = path
    return index


def _find_paths(area: str, path_index: dict[str, Path]) -> dict[str, Path]:
    try:
        return {band: path_index[f"{area}_{band}.tif"] for band in OPTICAL_BANDS}
    except KeyError:
        pass
    raise FileNotFoundError(f"real twelve-band S2 source unavailable for {area}")


def _canonical_s2(paths: dict[str, Path]) -> np.ndarray:
    with rasterio.open(paths["B02"]) as grid:
        cube = np.stack([_read_to_grid(paths[band], grid, Resampling.bilinear) for band in OPTICAL_BANDS])
    if cube.shape != (12, 120, 120) or not np.isfinite(cube).all():
        raise ValueError("canonical S2 cube failed validation")
    return _robust_channel_scale(cube).astype(np.float32)


def _select_manifest(records: list[dict], roots: list[Path], path_index: dict[str, Path], *, split: str) -> list[dict]:
    selected = []
    for record in records:
        paths = _find_paths(record["image_id"], path_index)
        selected.append({
            "record_id": record["record_id"],
            "annotation_id": record["annotation_id"],
            "image_id": record["image_id"],
            "split": split,
            "task_type": record["task_type"],
            "question": record["question"],
            "answer": record["answer"],
            "source_root": str(next(root for root in roots if paths["B02"].is_relative_to(root))),
            "s2_source_hash": _area_source_hash(paths),
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--s2-root", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, default=Path("artifacts/annotations/image_language/visual_vqa_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3g"))
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--validation-limit", type=int, default=50)
    args = parser.parse_args()
    if args.train_limit < 1 or args.validation_limit < 1:
        raise ValueError("pilot limits must be positive")
    config = PilotConfig(train_limit=args.train_limit, validation_limit=args.validation_limit)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_PHASE3G_PILOT")
    if not args.model_snapshot.is_dir():
        raise FileNotFoundError(args.model_snapshot)
    roots = [path.resolve() for path in args.s2_root]
    path_index = _build_path_index(roots)
    names = set(path_index)
    train_records = _load_records(args.annotations, names, split="train", limit=config.train_limit)
    validation_records = _load_records(args.annotations, names, split="validation", limit=config.validation_limit)
    selection = {
        "train": _select_manifest(train_records, roots, path_index, split="train"),
        "validation": _select_manifest(validation_records, roots, path_index, split="validation"),
    }
    manifest_basis = {
        "seed": config.seed,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "annotation_fingerprint": ANNOTATION_FINGERPRINT,
        "annotation_source_revision": ANNOTATION_SOURCE_REVISION,
        "representation": "canonical_real_s2_12x120x120_robust_channel_scale_v1",
        "scientific_representation_used": False,
        "selection": selection,
    }
    manifest_basis["representation_fingerprint"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_basis["unique_image_counts"] = {
        split: len({record["image_id"] for record in values}) for split, values in selection.items()
    }
    manifest_basis["duplicate_record_counts"] = {
        split: len(values) - len({record["record_id"] for record in values}) for split, values in selection.items()
    }
    manifest_basis["cross_split_image_intersection"] = len(
        {record["image_id"] for record in selection["train"]}
        & {record["image_id"] for record in selection["validation"]}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_manifest.json").write_text(json.dumps(manifest_basis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    adapter = Qwen25VLRGBAdapter()
    runtime = adapter.load_model(args.model_snapshot, dtype=config.precision, device="cuda")
    model = runtime["model"]
    model.eval()
    parameter_counts = freeze_qwen(model)
    projector = S2MultispectralProjector().to(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    process = psutil.Process()
    metrics = {"train_loss": [], "validation_loss": []}
    peak_allocated = 0
    peak_reserved = 0
    peak_ram = process.memory_info().rss
    started = time.perf_counter()
    train_started = time.perf_counter()
    answer_tokens = {"train": 0, "validation": 0}

    def run_record(record: dict, *, training: bool) -> float:
        nonlocal peak_allocated, peak_reserved, peak_ram
        paths = _find_paths(record["image_id"], path_index)
        s2 = torch.from_numpy(_canonical_s2(paths)).unsqueeze(0)
        batch = build_qwen_token_batch(
            tokenizer=runtime["processor"].tokenizer,
            model=model,
            projector=projector,
            s2=s2,
            questions=[record["question"]],
            answers=[record["answer"]],
        )
        with torch.set_grad_enabled(training):
            output = model(**batch)
            loss = output.loss
            if training:
                (loss / config.gradient_accumulation_steps).backward()
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
        peak_ram = max(peak_ram, process.memory_info().rss)
        answer_tokens["train" if training else "validation"] += int(batch["labels"].ne(-100).sum().item())
        return float(loss.detach().cpu())

    for index, record in enumerate(selection["train"]):
        if index % config.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        metrics["train_loss"].append(run_record(record, training=True))
        if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == len(selection["train"]):
            optimizer.step()
            projector.zero_grad(set_to_none=True)
    train_elapsed = time.perf_counter() - train_started
    validation_started = time.perf_counter()
    with torch.no_grad():
        for record in selection["validation"]:
            metrics["validation_loss"].append(run_record(record, training=False))
    validation_elapsed = time.perf_counter() - validation_started
    summary = {
        "status": "PASS",
        "pilot_type": "proof_of_concept_not_benchmark",
        "train_count": len(selection["train"]),
        "validation_count": len(selection["validation"]),
        "test_count": 0,
        "seed": config.seed,
        "device": "cuda",
        "dtype": config.precision,
        "quantization": config.quantization,
        "model_revision": DEFAULT_MODEL_REVISION,
        "model_load_seconds": runtime["load_seconds"],
        "elapsed_seconds": time.perf_counter() - started,
        "train_seconds": train_elapsed,
        "validation_seconds": validation_elapsed,
        "train_answer_tokens": answer_tokens["train"],
        "validation_answer_tokens": answer_tokens["validation"],
        "train_answer_tokens_per_second": answer_tokens["train"] / train_elapsed,
        "validation_answer_tokens_per_second": answer_tokens["validation"] / validation_elapsed,
        "unique_train_images": manifest_basis["unique_image_counts"]["train"],
        "unique_validation_images": manifest_basis["unique_image_counts"]["validation"],
        "cross_split_image_intersection": manifest_basis["cross_split_image_intersection"],
        "trainable_parameters": projector.trainable_parameter_count,
        **parameter_counts,
        "train_loss_mean": float(np.mean(metrics["train_loss"])),
        "validation_loss_mean": float(np.mean(metrics["validation_loss"])),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_ram_bytes": peak_ram,
        "metrics": metrics,
    }
    provenance = projector.provenance(
        model_id=adapter.model_id,
        checkpoint_revision=DEFAULT_MODEL_REVISION,
        input_shape=(1, 12, 120, 120),
    )
    provenance.update({
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "annotation_fingerprint": ANNOTATION_FINGERPRINT,
        "representation_fingerprint": manifest_basis["representation_fingerprint"],
        "training_seed": config.seed,
        "device": "cuda",
        "dtype": config.precision,
        "quantization": config.quantization,
    })
    checkpoint = args.output_dir / "s2_multispectral_projector.pt"
    summary["checkpoint_sha256"] = save_projector_checkpoint(
        checkpoint, projector, provenance=provenance, config=config, training_summary=summary,
    )
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
