"""Fail-closed data contracts for the Phase 3O adaptation pilot.

The contract deliberately accepts only the validated BigEarthNet.txt language
tasks.  It is not a benchmark reader and it never treats ``hybrid_830d`` as a
language-model input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SAFE_TASKS = frozenset({"VQA_BINARY", "VQA_MULTIPLE_CHOICE", "CAPTIONING"})
TRAINING_SPLITS = frozenset({"train", "validation"})
MODE_RGB = "MODE_RGB"
MODE_S2_PROJECTED = "MODE_S2_PROJECTED"
SUPPORTED_MODES = frozenset({MODE_RGB, MODE_S2_PROJECTED})


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class TrainingRecord:
    """A traceable, split-safe supervision record for a language adaptation."""

    sample_id: str
    image_id: str
    split: str
    task_type: str
    input_representation: str
    question: str | None
    answer: str | None
    caption: str | None
    source_annotation_revision: str
    representation_fingerprint: str
    provenance: Mapping[str, Any]
    license_status: str
    augmentation_policy: str

    def validate(self) -> None:
        if not self.sample_id or not self.image_id:
            raise ValueError("sample_id and image_id are required")
        if self.split not in TRAINING_SPLITS:
            raise ValueError("test records and unknown splits are forbidden")
        if self.task_type not in SAFE_TASKS:
            raise ValueError("only validated binary, multiple-choice, and caption tasks are allowed")
        if self.input_representation not in SUPPORTED_MODES:
            raise ValueError("unsupported input representation")
        if self.input_representation == MODE_RGB and self.provenance.get("representation") != "rgb_image_to_qwen":
            raise ValueError("MODE_RGB requires explicit RGB provenance")
        if self.input_representation == MODE_S2_PROJECTED:
            if self.provenance.get("representation") != "raw_s2_12x120x120_to_learned_qwen_tokens":
                raise ValueError("MODE_S2_PROJECTED requires raw-S2 projector provenance")
            if self.provenance.get("canonical_band_order") != ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]:
                raise ValueError("MODE_S2_PROJECTED requires the canonical 12-band order")
        if self.task_type == "CAPTIONING":
            if self.caption is None or self.answer is not None or self.question is not None:
                raise ValueError("caption records must supervise caption only")
        elif not self.question or not self.answer or self.caption is not None:
            raise ValueError("VQA records must supervise question and answer only")
        if not self.source_annotation_revision or len(self.representation_fingerprint) != 64:
            raise ValueError("source revision and representation fingerprint are required")
        if self.license_status != "PROJECT_AUTHORIZED_BIGEARTHNET_TXT":
            raise ValueError("training data must be explicitly project-authorized")
        if self.augmentation_policy != "NONE_PHASE3O_PILOT":
            raise ValueError("unrecorded augmentation is forbidden")
        if self.provenance.get("scientific_representation_used") is not False:
            raise ValueError("scientific representations are forbidden from adaptation")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def record_from_sample(sample: Mapping[str, Any], *, mode: str, raw_representation: Mapping[str, Any]) -> TrainingRecord:
    """Create a record after an image asset has been independently resolved."""
    if sample.get("validation_status") != "VALID" or sample.get("task_type") not in SAFE_TASKS:
        raise ValueError("sample is outside the validated Phase 3O task boundary")
    if sample.get("split") not in TRAINING_SPLITS:
        raise ValueError("test samples are forbidden")
    source = sample.get("source_provenance") or {}
    if mode == MODE_S2_PROJECTED:
        provenance = {
            "representation": "raw_s2_12x120x120_to_learned_qwen_tokens",
            "canonical_band_order": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
            "raw_source_sha256": raw_representation["raw_source_sha256"],
            "preprocessing": "robust_channel_scale_v1",
            "projector_interface": "[1,12,120,120]->[1,16,2048]",
            "scientific_representation_used": False,
        }
    elif mode == MODE_RGB:
        provenance = {
            "representation": "rgb_image_to_qwen",
            "raw_source_sha256": raw_representation["raw_source_sha256"],
            "preprocessing": "qwen_rgb_processor",
            "scientific_representation_used": False,
        }
    else:
        raise ValueError("unsupported input representation")
    fingerprint = sha256(canonical_json({"image_id": sample["image_id"], "mode": mode, "provenance": provenance})).hexdigest()
    record = TrainingRecord(
        sample_id=sample["sample_id"], image_id=sample["image_id"], split=sample["split"], task_type=sample["task_type"],
        input_representation=mode, question=sample.get("question"), answer=sample.get("answer"), caption=sample.get("caption"),
        source_annotation_revision=source["source_revision"], representation_fingerprint=fingerprint, provenance=provenance,
        license_status="PROJECT_AUTHORIZED_BIGEARTHNET_TXT", augmentation_policy="NONE_PHASE3O_PILOT",
    )
    record.validate()
    return record


def load_safe_records(samples_path: Path, *, mode: str, raw_sources: Mapping[str, Mapping[str, Any]], limits: Mapping[str, int]) -> dict[str, list[TrainingRecord]]:
    """Source-order selection, requiring a verified local raw asset for each record."""
    selected = {"train": [], "validation": []}
    with Path(samples_path).open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            split = sample.get("split")
            if split not in selected or len(selected[split]) >= limits[split] or sample.get("image_id") not in raw_sources:
                continue
            if sample.get("task_type") in SAFE_TASKS and sample.get("validation_status") == "VALID":
                selected[split].append(record_from_sample(sample, mode=mode, raw_representation=raw_sources[sample["image_id"]]))
    if any(len(selected[split]) != limits[split] for split in selected):
        raise RuntimeError("INSUFFICIENT_VERIFIED_LOCAL_RAW_ASSETS")
    validate_record_partitions(selected)
    return selected


def validate_record_partitions(records: Mapping[str, Iterable[TrainingRecord]]) -> None:
    train = list(records.get("train", ()))
    validation = list(records.get("validation", ()))
    for record in [*train, *validation]:
        record.validate()
    train_ids, validation_ids = {r.image_id for r in train}, {r.image_id for r in validation}
    if train_ids & validation_ids:
        raise ValueError("cross-split image leakage")
    if any(r.split != "train" for r in train) or any(r.split != "validation" for r in validation):
        raise ValueError("record is in the wrong training partition")


def checkpoint_manifest(*, checkpoint_path: Path | None, provenance: Mapping[str, Any], status: str) -> dict[str, Any]:
    """Describe adapter-only checkpoint state without serializing a Qwen copy."""
    path = Path(checkpoint_path) if checkpoint_path else None
    digest = sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
    return {"status": status, "adapter_checkpoint": path.name if path else None, "checkpoint_sha256": digest,
            "base_model_weights_saved": False, "provenance": dict(provenance), "scientific_representation_used": False}
