from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from src.annotation_foundation import canonical_bytes
from src.image_language_foundation import (
    AdapterOperation, BigEarthNetTextAdapter, CDVQAAdapter, ImageLanguageSample,
    InputModality, LanguageTaskType, ModelOutputType, PromptTemplate,
    RepresentationInput, VLMCapabilities, VLMAdapter, assemble_sample,
    build_bigearthnet_text_view, representation_input_from_catalog,
    iter_dataset_view,
)


def annotation(**changes):
    value = {
        "schema_version": "annotation_schema_v1", "annotation_id": "ann-1", "image_id": "area-1",
        "split": "train", "task_type": "binary_qa", "question": "Is café water visible?",
        "answer": "yes", "choices": None, "options": None, "caption": None,
        "box": None, "point": None, "geometry_frame": None, "coordinate_convention_status": "UNKNOWN",
        "validation_status": "validated", "source_dataset": "BigEarthNet.txt", "source_record_id": "1",
        "source_record_sha256": "a" * 64, "source_task_type": "binary", "source_category": "presence",
        "source_partition": "train", "provenance": {"source_revision": "pinned"},
        "raw_source": {"input": "Is café water visible?"},
    }
    value.update(changes)
    return value


def representation(name, dimension, **changes):
    value = RepresentationInput(
        reference_id=f"area-1:{name}:default:abc", representation_type=name, variant="default",
        modality=InputModality.OPTICAL_SAR, shape=(dimension,), dtype="float32",
        artifact_uri=f"artifact://pipeline3/shard/{name}", checksum_sha256="b" * 64,
        sample_sha256="c" * 64, producer="fixture", model_version=None,
        preprocessing_version="v1", spatial_level="scene", validation_status="VERIFIED",
    )
    return replace(value, **changes)


def core():
    return (representation("physical_62d", 62), representation("joint_croma_gap_768d", 768),
            representation("hybrid_830d", 830))


def test_valid_sample_text_unicode_choices_and_hybrid_distinction():
    source = annotation(task_type="multiple_choice_qa", answer="b",
                        choices=[{"key": "a", "text": "Forêt"}, {"key": "b", "text": "Water"}])
    sample = assemble_sample(source, {"split": "train", "dataset_fingerprint": "fp"}, core())
    assert sample.task_type is LanguageTaskType.VQA_MULTIPLE_CHOICE
    assert sample.question == source["question"] and sample.answer == "b"
    assert sample.choices == tuple(source["choices"])
    assert sample.source_text == "Is café water visible?"
    assert sample.input_profile["primary_visual"] == "joint_croma_gap_768d"
    assert sample.input_profile["hybrid_is_default_vlm_input"] is False
    assert {ref.representation_type: ref.shape for ref in sample.representation_refs} == {
        "physical_62d": (62,), "joint_croma_gap_768d": (768,), "hybrid_830d": (830,)}


@pytest.mark.parametrize("source,image,refs,match", [
    (None, {"split": "train"}, (), "missing annotation"),
    (annotation(), None, core(), "missing image"),
    (annotation(), {"split": "test"}, core(), "split mismatch"),
    (annotation(), {"split": "train"}, core()[:2], "missing required representation"),
    (annotation(), {"split": "train"}, tuple(replace(r, reference_id=r.reference_id.replace("area-1", "area-2")) for r in core()), "provenance mismatch"),
])
def test_fail_closed_assembly(source, image, refs, match):
    with pytest.raises(ValueError, match=match):
        assemble_sample(source, image, refs)


def test_task_modality_and_unverified_representation_validation():
    with pytest.raises(ValueError, match="unsupported annotation task"):
        assemble_sample(annotation(task_type="temporal_vqa"), {"split": "train"}, core())
    row = {
        "availability": "VERIFIED", "validation_status": "VERIFIED", "representation_id": "r",
        "representation_type": "joint", "artifact_ref": {}, "provenance": {}, "modality": "unknown",
        "shape": [1], "dtype": "float32", "logical_uri": "artifact://x/y", "checksum": "a" * 64,
        "producer": "x", "model_version": None, "preprocessing_version": None, "spatial_semantics": {"level": "scene"},
    }
    with pytest.raises(ValueError, match="unsupported representation modality"):
        representation_input_from_catalog(row)
    row["modality"] = "optical_sar"; row["availability"] = "MISSING"
    with pytest.raises(ValueError, match="unverified"):
        representation_input_from_catalog(row)


def test_spatial_source_is_preserved_and_never_mapped():
    source = annotation(task_type="text_box", question=None, answer=None, box=[0.1, 0.2, 0.7, 0.8],
                        geometry_frame="source_unit_square_unmapped", referenced_text="forest")
    sample = assemble_sample(source, {"split": "train"}, core())
    assert sample.spatial_reference["status"] == "UNMAPPED"
    assert sample.spatial_reference["source_geometry"]["box"] == source["box"]
    assert sample.spatial_reference["analysis_geometry"] is None
    assert sample.spatial_reference["token_coordinates"] is None
    assert sample.spatial_reference["learned_grounding"] is False


def test_prompt_is_derived_and_adapter_interfaces_are_capability_explicit():
    sample = assemble_sample(annotation(), {"split": "train"}, core())
    template = PromptTemplate("binary-minimal", "v1", LanguageTaskType.VQA_BINARY,
                              ("question",), ModelOutputType.CATEGORICAL_ANSWER, "Question: {question}")
    derived = template.render(sample)
    assert derived["derived"] and derived["prompt"] == "Question: Is café water visible?"
    assert sample.question == "Is café water visible?"

    class Stub(VLMAdapter):
        def capabilities(self): return VLMCapabilities((AdapterOperation.SCORE,), supports_vqa=True)
        def provenance(self): return {"adapter": "stub"}
        def validate_inputs(self, sample): return None
        def prepare_inputs(self, sample): return sample.sample_id
    stub = Stub()
    assert stub.capabilities().supports_vqa and not stub.capabilities().supports_captioning
    with pytest.raises(NotImplementedError): stub.generate(stub.prepare_inputs(sample))
    with pytest.raises(NotImplementedError): CDVQAAdapter().assemble({}, {}, ())


def _write_dataset_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = [annotation(annotation_id="ann-1"), annotation(annotation_id="ann-2", task_type="caption",
            question=None, answer=None, caption="A forest scene."),
            annotation(annotation_id="ann-q", validation_status="quarantined")]
    content = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    annotations = root / "annotations.jsonl"; annotations.write_bytes(content)
    annotation_manifest = root / "annotation_manifest.json"
    annotation_manifest.write_text(json.dumps({
        "annotations_sha256": hashlib.sha256(content).hexdigest(), "schema_version": "annotation_schema_v1",
        "dataset_fingerprint": "dataset-fp", "split_fingerprint": "split-fp",
    }), encoding="utf-8")
    dataset = root / "dataset.json"
    dataset.write_text(json.dumps({"dataset_fingerprint": "dataset-fp", "rows": [{"area_id": "area-1", "split": "train"}]}), encoding="utf-8")
    catalog = root / "catalog.jsonl"
    catalog_rows = []
    for ref in core():
        catalog_rows.append({
            "image_id": "area-1", "availability": "VERIFIED", "validation_status": "VERIFIED",
            "representation_id": ref.reference_id, "representation_type": ref.representation_type,
            "artifact_ref": {"variant": "default", "sample_sha256": ref.sample_sha256},
            "provenance": {}, "modality": "optical_sar", "shape": list(ref.shape), "dtype": ref.dtype,
            "logical_uri": ref.artifact_uri, "checksum": ref.checksum_sha256, "producer": ref.producer,
            "model_version": None, "preprocessing_version": "v1", "spatial_semantics": {"level": "scene"},
        })
    catalog.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in catalog_rows))
    catalog_report = root / "catalog_report.json"; catalog_report.write_text(json.dumps({"catalog_sha256": "d" * 64}), encoding="utf-8")
    links = root / "links.json"; links.write_text(json.dumps({"links_sha256": "e" * 64}), encoding="utf-8")
    return annotations, annotation_manifest, dataset, catalog, catalog_report, links


def test_dataset_views_grouping_provenance_and_byte_determinism(tmp_path):
    paths = _write_dataset_fixture(tmp_path / "source")
    first = build_bigearthnet_text_view(*paths, tmp_path / "first", expected_sample_count=2, expected_image_count=1)
    second = build_bigearthnet_text_view(*paths, tmp_path / "second", expected_sample_count=2, expected_image_count=1)
    assert (tmp_path / "first/samples.jsonl").read_bytes() == (tmp_path / "second/samples.jsonl").read_bytes()
    assert (tmp_path / "first/manifest.json").read_bytes() == (tmp_path / "second/manifest.json").read_bytes()
    assert first["manifest"]["image_language_fingerprint"] == second["manifest"]["image_language_fingerprint"]
    assert first["manifest"]["representation_catalog_fingerprint"] == "d" * 64
    assert first["manifest"]["representation_link_fingerprint"] == "e" * 64
    assert first["report"]["samples_by_split"] == {"train": 2}
    assert first["report"]["grouping_unit"] == "image_id"
    assert first["report"]["annotation_row_random_split"] is False
    assert first["report"]["skipped_quarantined_annotations"] == 1
    assert [row["annotation_id"] for row in iter_dataset_view(tmp_path / "first/samples.jsonl", "captioning")] == ["ann-2"]
    assert len(list(iter_dataset_view(tmp_path / "first/samples.jsonl", "train"))) == 2
    with pytest.raises(ValueError, match="unsupported"):
        list(iter_dataset_view(tmp_path / "first/samples.jsonl", "random_rows"))
