"""Model-agnostic image-language contracts and deterministic dataset assembly.

No model inference, prompt-driven generation, learned grounding, or benchmark
evaluation occurs here.  The module joins existing annotation, image, split,
representation, linkage, and provenance contracts without changing them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .annotation_foundation import canonical_bytes, file_digest
from .image_language_dataset import parse_mcq_options


SCHEMA_VERSION = "image_language_sample_v1"
BUILD_VERSION = "image_language_foundation_v1"
CORE_REPRESENTATIONS = ("physical_62d", "joint_croma_gap_768d", "hybrid_830d")


class LanguageTaskType(str, Enum):
    VQA_BINARY = "VQA_BINARY"
    VQA_MULTIPLE_CHOICE = "VQA_MULTIPLE_CHOICE"
    CAPTIONING = "CAPTIONING"
    GROUNDING_TEXT_BOX = "GROUNDING_TEXT_BOX"
    GROUNDING_POINT = "GROUNDING_POINT"
    TEMPORAL_VQA = "TEMPORAL_VQA"
    CHANGE_DESCRIPTION = "CHANGE_DESCRIPTION"
    CHANGE_GROUNDING = "CHANGE_GROUNDING"


IMPLEMENTED_SOURCE_TASKS = frozenset({
    LanguageTaskType.VQA_BINARY, LanguageTaskType.VQA_MULTIPLE_CHOICE,
    LanguageTaskType.CAPTIONING, LanguageTaskType.GROUNDING_TEXT_BOX,
    LanguageTaskType.GROUNDING_POINT,
})
TASK_MAPPING = {
    "binary_qa": LanguageTaskType.VQA_BINARY,
    "multiple_choice_qa": LanguageTaskType.VQA_MULTIPLE_CHOICE,
    "caption": LanguageTaskType.CAPTIONING,
    "text_box": LanguageTaskType.GROUNDING_TEXT_BOX,
    "point_box": LanguageTaskType.GROUNDING_POINT,
}


class InputModality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    OPTICAL_SAR = "OPTICAL_SAR"


class ModelOutputType(str, Enum):
    FREE_TEXT = "free_text"
    CATEGORICAL_ANSWER = "categorical_answer"
    MULTIPLE_CHOICE_ANSWER = "multiple_choice_answer"
    STRUCTURED_ANSWER = "structured_answer"
    GROUNDING_OUTPUT = "grounding_output"


class AdapterOperation(str, Enum):
    GENERATE = "generate"
    SCORE = "score"
    GROUND = "ground"


@dataclass(frozen=True)
class RepresentationInput:
    reference_id: str
    representation_type: str
    variant: str
    modality: InputModality
    shape: tuple[int, ...]
    dtype: str
    artifact_uri: str
    checksum_sha256: str
    sample_sha256: str | None
    producer: str
    model_version: str | None
    preprocessing_version: str | None
    spatial_level: str
    validation_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", InputModality(self.modality))
        if self.validation_status != "VERIFIED":
            raise ValueError("language inputs require VERIFIED representations")
        if not self.reference_id or not self.representation_type or not self.variant:
            raise ValueError("representation input identity is required")
        if not self.shape or any(type(value) is not int or value <= 0 for value in self.shape):
            raise ValueError("representation input shape is invalid")
        for checksum in (self.checksum_sha256, self.sample_sha256):
            if checksum is not None and (len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum.lower())):
                raise ValueError("representation checksums must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modality"] = self.modality.value
        return value


@dataclass(frozen=True)
class VisionInputContract:
    available_modalities: tuple[InputModality, ...]
    available_representations: tuple[RepresentationInput, ...]
    spatial_resolution: Mapping[str, Any] | None
    feature_dimensionality: Mapping[str, int]
    provenance: Mapping[str, Any]
    preprocessing: Mapping[str, Any]

    def __post_init__(self) -> None:
        modalities = tuple(InputModality(value) for value in self.available_modalities)
        if len(set(modalities)) != len(modalities):
            raise ValueError("available modalities contain duplicates")
        object.__setattr__(self, "available_modalities", modalities)
        if any(not isinstance(value, RepresentationInput) for value in self.available_representations):
            raise ValueError("available representations must be typed inputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_modalities": [value.value for value in self.available_modalities],
            "available_representations": [value.to_dict() for value in self.available_representations],
            "spatial_resolution": self.spatial_resolution,
            "feature_dimensionality": dict(self.feature_dimensionality),
            "provenance": dict(self.provenance),
            "preprocessing": dict(self.preprocessing),
        }


@dataclass(frozen=True)
class ImageLanguageSample:
    sample_id: str
    image_id: str
    split: str
    task_type: LanguageTaskType
    annotation_id: str
    input_modalities: tuple[InputModality, ...]
    representation_refs: tuple[RepresentationInput, ...]
    input_profile: Mapping[str, Any]
    question: str | None
    answer: str | None
    choices: tuple[Mapping[str, Any], ...] | None
    caption: str | None
    source_text: str | None
    spatial_reference: Mapping[str, Any] | None
    source_dataset: str
    source_provenance: Mapping[str, Any]
    validation_status: str
    validation_issues: tuple[str, ...]
    scene_id: str | None = None
    observation_id: str | None = None
    temporal_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_type", LanguageTaskType(self.task_type))
        object.__setattr__(self, "input_modalities", tuple(InputModality(value) for value in self.input_modalities))
        if self.task_type not in IMPLEMENTED_SOURCE_TASKS:
            raise ValueError("reserved temporal task is not supported by the current source adapter")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("sample split is invalid")
        if not all((self.sample_id, self.image_id, self.annotation_id, self.source_dataset)):
            raise ValueError("sample identity is incomplete")
        if self.validation_status != "VALID" or self.validation_issues:
            raise ValueError("canonical samples must pass fail-closed validation")
        if not self.representation_refs:
            raise ValueError("sample has no verified representation")
        if any(value.validation_status != "VERIFIED" for value in self.representation_refs):
            raise ValueError("sample contains an unverified representation")
        if self.task_type in {LanguageTaskType.VQA_BINARY, LanguageTaskType.VQA_MULTIPLE_CHOICE}:
            if not isinstance(self.question, str) or not self.question or not isinstance(self.answer, str) or not self.answer:
                raise ValueError("VQA source records require question and answer text")
        if self.task_type is LanguageTaskType.VQA_MULTIPLE_CHOICE and not self.choices:
            raise ValueError("multiple-choice source record has no choices")
        if self.task_type is LanguageTaskType.CAPTIONING and (not isinstance(self.caption, str) or not self.caption):
            raise ValueError("captioning source record has no caption")
        if any(value is not None for value in (self.scene_id, self.observation_id, self.temporal_group_id)):
            raise ValueError("temporal identities are unavailable for BigEarthNet.txt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "sample_id": self.sample_id, "image_id": self.image_id,
            "split": self.split, "task_type": self.task_type.value, "annotation_id": self.annotation_id,
            "input_modalities": [value.value for value in self.input_modalities],
            "representation_refs": [
                {"reference_id": value.reference_id, "representation_type": value.representation_type,
                 "variant": value.variant, "modality": value.modality.value}
                for value in self.representation_refs
            ],
            "input_profile": dict(self.input_profile), "question": self.question, "answer": self.answer,
            "choices": list(self.choices) if self.choices is not None else None, "caption": self.caption,
            "source_text": self.source_text, "spatial_reference": self.spatial_reference,
            "source_dataset": self.source_dataset, "source_provenance": dict(self.source_provenance),
            "validation_status": self.validation_status, "validation_issues": list(self.validation_issues),
            "scene_id": self.scene_id, "observation_id": self.observation_id,
            "temporal_group_id": self.temporal_group_id,
        }


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    version: str
    task_type: LanguageTaskType
    input_fields: tuple[str, ...]
    output_format: ModelOutputType
    template: str

    def render(self, sample: ImageLanguageSample) -> dict[str, Any]:
        if sample.task_type is not self.task_type:
            raise ValueError("prompt template task mismatch")
        values = {field: getattr(sample, field) for field in self.input_fields}
        if any(value is None for value in values.values()):
            raise ValueError("prompt input field is unavailable")
        text = self.template.format(**values)
        return {"derived": True, "template_id": self.template_id, "template_version": self.version,
                "prompt": text, "source_annotation_id": sample.annotation_id}


@dataclass(frozen=True)
class GroundingOutput:
    phrase: str
    region_id: str | None
    bbox: tuple[float, float, float, float] | None
    mask_ref: str | None
    polygon: tuple[tuple[float, float], ...] | None
    confidence: float | None
    provenance: Mapping[str, Any]
    learned_prediction: bool = True


@dataclass(frozen=True)
class ModelOutput:
    output_type: ModelOutputType
    value: Any
    adapter_id: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class VLMCapabilities:
    operations: tuple[AdapterOperation, ...]
    supports_vqa: bool = False
    supports_captioning: bool = False
    supports_grounding: bool = False
    supports_image_only: bool = False
    supports_multi_image: bool = False
    supports_optical_sar: bool = False


class VisionInputAdapter(ABC):
    @abstractmethod
    def validate_inputs(self, sample: ImageLanguageSample) -> None: ...

    @abstractmethod
    def prepare_inputs(self, sample: ImageLanguageSample) -> Any: ...

    @abstractmethod
    def contract(self, sample: ImageLanguageSample) -> VisionInputContract: ...


class VLMAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> VLMCapabilities: ...

    @abstractmethod
    def provenance(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def validate_inputs(self, sample: ImageLanguageSample) -> None: ...

    @abstractmethod
    def prepare_inputs(self, sample: ImageLanguageSample) -> Any: ...

    def generate(self, prepared_inputs: Any) -> ModelOutput:
        raise NotImplementedError("adapter does not implement generation")

    def score(self, prepared_inputs: Any, candidates: Sequence[str]) -> ModelOutput:
        raise NotImplementedError("adapter does not implement scoring")


class LanguageTaskEvaluator(ABC):
    """Future metric boundary; no benchmark metrics are implemented in Phase 2E."""

    @abstractmethod
    def supported_tasks(self) -> tuple[LanguageTaskType, ...]: ...

    @abstractmethod
    def evaluate(self, predictions: Iterable[ModelOutput], samples: Iterable[ImageLanguageSample]) -> Mapping[str, Any]: ...


class BenchmarkAdapter(ABC):
    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    def assemble(self, annotation: Mapping[str, Any], image: Mapping[str, Any],
                 representations: Sequence[RepresentationInput]) -> ImageLanguageSample: ...


class BigEarthNetTextAdapter(BenchmarkAdapter):
    def source_name(self) -> str:
        return "BigEarthNet.txt"

    def assemble(self, annotation: Mapping[str, Any], image: Mapping[str, Any],
                 representations: Sequence[RepresentationInput]) -> ImageLanguageSample:
        return assemble_sample(annotation, image, representations)


class RSVQAAdapter(BenchmarkAdapter):
    def source_name(self) -> str: return "RSVQA"
    def assemble(self, annotation, image, representations):
        raise NotImplementedError("RSVQA is a reserved future adapter")


class VRSBenchAdapter(BenchmarkAdapter):
    def source_name(self) -> str: return "VRSBench"
    def assemble(self, annotation, image, representations):
        raise NotImplementedError("VRSBench is a reserved future adapter")


class CDVQAAdapter(BenchmarkAdapter):
    def source_name(self) -> str: return "CDVQA"
    def assemble(self, annotation, image, representations):
        raise NotImplementedError("CDVQA is a reserved future adapter")


def representation_input_from_catalog(row: Mapping[str, Any]) -> RepresentationInput:
    if row.get("availability") != "VERIFIED" or row.get("validation_status") != "VERIFIED":
        raise ValueError("unverified catalog representation")
    modality = {"optical": InputModality.OPTICAL, "sar": InputModality.SAR,
                "optical_sar": InputModality.OPTICAL_SAR}.get(row.get("modality"))
    if modality is None:
        raise ValueError("unsupported representation modality")
    artifact = row.get("artifact_ref") or {}
    provenance = row.get("provenance") or {}
    return RepresentationInput(
        reference_id=row["representation_id"], representation_type=row["representation_type"],
        variant=artifact.get("variant", provenance.get("variant", "default")), modality=modality,
        shape=tuple(row["shape"]), dtype=row["dtype"], artifact_uri=row["logical_uri"],
        checksum_sha256=row["checksum"], sample_sha256=artifact.get("sample_sha256", provenance.get("sample_sha256")),
        producer=row["producer"], model_version=row.get("model_version"),
        preprocessing_version=row.get("preprocessing_version"),
        spatial_level=(row.get("spatial_semantics") or {}).get("level", "scene"),
        validation_status=row["validation_status"],
    )


def assemble_sample(annotation: Mapping[str, Any] | None, image: Mapping[str, Any] | None,
                    representations: Sequence[RepresentationInput]) -> ImageLanguageSample:
    if not isinstance(annotation, Mapping):
        raise ValueError("missing annotation")
    if not isinstance(image, Mapping):
        raise ValueError("missing image")
    annotation_id, image_id = annotation.get("annotation_id"), annotation.get("image_id")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError("missing annotation identity")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("missing image identity")
    split = annotation.get("split")
    if split != image.get("split"):
        raise ValueError("annotation/image split mismatch")
    if annotation.get("validation_status") != "validated":
        raise ValueError("annotation is not validated")
    try:
        task = TASK_MAPPING[annotation.get("task_type")]
    except KeyError as exc:
        raise ValueError("unsupported annotation task type") from exc
    refs = tuple(sorted(representations, key=lambda value: value.representation_type))
    by_type = {value.representation_type: value for value in refs}
    if len(by_type) != len(refs):
        raise ValueError("duplicate representation type")
    missing = set(CORE_REPRESENTATIONS).difference(by_type)
    if missing:
        raise ValueError(f"missing required representation: {sorted(missing)}")
    if any(value.validation_status != "VERIFIED" for value in refs):
        raise ValueError("representation provenance is unverified")
    dataset_fingerprint = image.get("dataset_fingerprint")
    if any(value.reference_id.split(":", 1)[0] != image_id for value in refs):
        raise ValueError("representation/image provenance mismatch")

    spatial = None
    if task in {LanguageTaskType.GROUNDING_TEXT_BOX, LanguageTaskType.GROUNDING_POINT}:
        geometry = {"box": annotation.get("box")} if task is LanguageTaskType.GROUNDING_TEXT_BOX else {"point": annotation.get("point")}
        spatial = {
            "status": "UNMAPPED", "source_geometry": geometry,
            "source_coordinate_space": annotation.get("geometry_frame"),
            "coordinate_convention_status": annotation.get("coordinate_convention_status"),
            "analysis_geometry": None, "token_coordinates": None,
            "geometric_correspondence": False, "learned_grounding": False,
        }
    raw_source = annotation.get("raw_source") or {}
    source_text = raw_source.get("input")
    if source_text is None:
        source_text = annotation.get("question") or annotation.get("caption") or annotation.get("referenced_text")
    raw_choices = annotation.get("choices") or annotation.get("options")
    choices_derivation = None
    if task is LanguageTaskType.VQA_MULTIPLE_CHOICE and raw_choices is None:
        raw_choices, parse_issue = parse_mcq_options(annotation.get("question"))
        if parse_issue is not None or raw_choices is None:
            raise ValueError(f"multiple-choice options unavailable: {parse_issue}")
        choices_derivation = "image_language_dataset.parse_mcq_options"
    choices = tuple(raw_choices) if isinstance(raw_choices, list) else None
    sample_id = "ils:" + hashlib.sha256(canonical_bytes([SCHEMA_VERSION, annotation_id])).hexdigest()
    return ImageLanguageSample(
        sample_id=sample_id, image_id=image_id, split=split, task_type=task, annotation_id=annotation_id,
        input_modalities=tuple(sorted({value.modality for value in refs}, key=lambda value: value.value)),
        representation_refs=refs,
        input_profile={
            "primary_visual": "joint_croma_gap_768d", "physical": "physical_62d",
            "scientific_predictor_input": "hybrid_830d", "hybrid_is_default_vlm_input": False,
            "raw_optical": "MISSING", "raw_sar": "MISSING",
        },
        question=annotation.get("question"), answer=annotation.get("answer"), choices=choices,
        caption=annotation.get("caption"), source_text=source_text, spatial_reference=spatial,
        source_dataset=annotation.get("source_dataset") or "BigEarthNet.txt",
        source_provenance={
            "annotation_schema_version": annotation.get("schema_version"),
            "source_revision": (annotation.get("provenance") or {}).get("source_revision"),
            "source_record_id": annotation.get("source_record_id"),
            "source_record_sha256": annotation.get("source_record_sha256"),
            "source_task_type": annotation.get("source_task_type"),
            "source_category": annotation.get("source_category"),
            "source_partition": annotation.get("source_partition"),
            "dataset_fingerprint": dataset_fingerprint,
            "choices_derivation": choices_derivation,
        },
        validation_status="VALID", validation_issues=(),
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_bytes(value) + b"\n")


def _load_verified_representations(path: Path) -> dict[str, tuple[RepresentationInput, ...]]:
    result: dict[str, list[RepresentationInput]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("representation_type") not in CORE_REPRESENTATIONS:
                continue
            if row.get("availability") != "VERIFIED":
                continue
            result.setdefault(row["image_id"], []).append(representation_input_from_catalog(row))
    return {key: tuple(value) for key, value in result.items()}


def iter_dataset_view(samples_path: Path, view: str) -> Iterable[dict[str, Any]]:
    """Stream a named logical view without copying or repartitioning samples."""
    split_views = {"train", "validation", "test"}
    task_views = {
        "binary_vqa": {"VQA_BINARY"}, "mcq_vqa": {"VQA_MULTIPLE_CHOICE"},
        "captioning": {"CAPTIONING"},
        "grounding": {"GROUNDING_TEXT_BOX", "GROUNDING_POINT"},
    }
    if view != "all" and view not in split_views and view not in task_views:
        raise ValueError("unsupported image-language dataset view")
    with Path(samples_path).open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if view == "all" or (view in split_views and sample["split"] == view) or (
                view in task_views and sample["task_type"] in task_views[view]
            ):
                yield sample


def build_bigearthnet_text_view(
    annotation_path: Path, annotation_manifest_path: Path, dataset_manifest_path: Path,
    representation_catalog_path: Path, representation_catalog_report_path: Path,
    representation_link_manifest_path: Path, output: Path,
    *, expected_sample_count: int = 101542, expected_image_count: int = 5000,
) -> dict[str, Any]:
    """Stream the canonical validated BigEarthNet.txt view in source order."""
    annotation_path, output = Path(annotation_path), Path(output)
    annotation_manifest = json.loads(Path(annotation_manifest_path).read_text(encoding="utf-8"))
    if file_digest(annotation_path) != annotation_manifest.get("annotations_sha256"):
        raise ValueError("annotation artifact checksum mismatch")
    dataset = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    images = {row["area_id"]: {"split": row["split"], "dataset_fingerprint": dataset.get("dataset_fingerprint")}
              for row in dataset.get("rows", [])}
    if len(images) != expected_image_count:
        raise ValueError(f"image manifest contains {len(images)}, expected {expected_image_count}")
    representations = _load_verified_representations(representation_catalog_path)
    catalog_report = json.loads(Path(representation_catalog_report_path).read_text(encoding="utf-8"))
    link_manifest = json.loads(Path(representation_link_manifest_path).read_text(encoding="utf-8"))

    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "samples.jsonl"
    descriptor, temporary = tempfile.mkstemp(prefix=".samples.", suffix=".tmp", dir=output)
    digest = hashlib.sha256()
    task_counts: Counter[str] = Counter(); split_counts: Counter[str] = Counter()
    image_splits: dict[str, str] = {}; seen_annotations: set[str] = set()
    validated_source = 0; skipped_quarantined = 0; spatial = 0; unmapped = 0
    missing_representation = 0; provenance_failures = 0; split_failures = 0; malformed = 0
    previous_annotation_id: str | None = None
    adapter = BigEarthNetTextAdapter()
    try:
        with os.fdopen(descriptor, "wb") as target, annotation_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    annotation = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed += 1
                    raise ValueError(f"malformed annotation line {line_number}") from exc
                if annotation.get("validation_status") != "validated":
                    skipped_quarantined += 1
                    continue
                validated_source += 1
                annotation_id = annotation.get("annotation_id")
                if annotation_id in seen_annotations:
                    raise ValueError(f"duplicate annotation: {annotation_id}")
                seen_annotations.add(annotation_id)
                if previous_annotation_id is not None and annotation_id <= previous_annotation_id:
                    raise ValueError("annotation source ordering is not deterministic annotation_id order")
                previous_annotation_id = annotation_id
                image_id = annotation.get("image_id")
                image = images.get(image_id)
                refs = representations.get(image_id, ())
                try:
                    sample = adapter.assemble(annotation, image, refs)
                except ValueError as exc:
                    message = str(exc)
                    missing_representation += "missing required representation" in message
                    provenance_failures += "provenance" in message
                    split_failures += "split mismatch" in message
                    raise
                content = canonical_bytes(sample.to_dict()) + b"\n"
                target.write(content); digest.update(content)
                task_counts[sample.task_type.value] += 1; split_counts[sample.split] += 1
                if image_id in image_splits and image_splits[image_id] != sample.split:
                    raise ValueError("cross-split image leakage")
                image_splits[image_id] = sample.split
                if sample.spatial_reference is not None:
                    spatial += 1
                    unmapped += sample.spatial_reference.get("status") == "UNMAPPED"
            target.flush(); os.fsync(target.fileno())
        os.replace(temporary, samples_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    if validated_source != expected_sample_count:
        raise ValueError(f"expected {expected_sample_count:,} validated annotations, got {validated_source}")
    if set(task_counts).difference(task.value for task in IMPLEMENTED_SOURCE_TASKS):
        raise ValueError("unsupported task entered canonical view")
    cross_split = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    report = {
        "schema_version": SCHEMA_VERSION, "build_version": BUILD_VERSION,
        "total_samples": validated_source, "skipped_quarantined_annotations": skipped_quarantined,
        "samples_by_task": dict(sorted(task_counts.items())), "samples_by_split": dict(sorted(split_counts.items())),
        "unique_images": len(image_splits), "samples_with_verified_core_representations": validated_source,
        "samples_missing_required_representation": missing_representation,
        "spatial_annotation_count": spatial, "deliberately_unmapped_spatial_count": unmapped,
        "provenance_failures": provenance_failures, "split_failures": split_failures,
        "malformed_records": malformed, "duplicate_records": 0,
        "cross_split_image_intersections": cross_split, "annotation_row_random_split": False,
        "grouping_unit": "image_id", "temporal_grouping": "reserved_not_populated",
    }
    views = {
        "all": {"filter": {}, "count": validated_source},
        **{name: {"filter": {"split": name}, "count": split_counts[name]} for name in ("train", "validation", "test")},
        "binary_vqa": {"filter": {"task_type": "VQA_BINARY"}, "count": task_counts["VQA_BINARY"]},
        "mcq_vqa": {"filter": {"task_type": "VQA_MULTIPLE_CHOICE"}, "count": task_counts["VQA_MULTIPLE_CHOICE"]},
        "captioning": {"filter": {"task_type": "CAPTIONING"}, "count": task_counts["CAPTIONING"]},
        "grounding": {"filter": {"task_type_in": ["GROUNDING_TEXT_BOX", "GROUNDING_POINT"]},
                      "count": task_counts["GROUNDING_TEXT_BOX"] + task_counts["GROUNDING_POINT"]},
    }
    manifest_basis = {
        "schema_version": SCHEMA_VERSION, "build_version": BUILD_VERSION,
        "sample_count": validated_source, "samples_file": "samples.jsonl", "samples_sha256": digest.hexdigest(),
        "ordering": "source annotation_id ascending", "annotation_source_sha256": annotation_manifest["annotations_sha256"],
        "annotation_schema_version": annotation_manifest["schema_version"],
        "dataset_fingerprint": annotation_manifest["dataset_fingerprint"],
        "split_fingerprint": annotation_manifest["split_fingerprint"],
        "representation_catalog_fingerprint": catalog_report["catalog_sha256"],
        "representation_link_fingerprint": link_manifest["links_sha256"], "views": views,
    }
    manifest = dict(manifest_basis, image_language_fingerprint=hashlib.sha256(canonical_bytes(manifest_basis)).hexdigest())
    provenance = {
        "schema_version": SCHEMA_VERSION, "annotation_artifact": "artifact://annotations/bigearthnet_txt/annotations.jsonl",
        "representation_catalog": "artifact://representation-catalog/pipeline3_5000/records.jsonl",
        "representation_links": "artifact://representation-links/bigearthnet_txt_phase2d2/links.jsonl",
        "machine_specific_paths_excluded": True, "model_training_performed": False,
        "vlm_inference_performed": False, "raw_imagery_copied": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "validation_report.json", report)
    _atomic_json(output / "provenance.json", provenance)
    return {"manifest": manifest, "report": report, "provenance": provenance}
