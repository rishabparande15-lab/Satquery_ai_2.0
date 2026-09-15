from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from src.architecture_contracts import (
    ArtifactRef,
    ArtifactType,
    RepresentationRef,
    RepresentationSet,
    RepresentationType,
    SceneBundle,
    SpatialReference,
    adapt_run_analysis_output,
)
from src.representation_artifacts import (
    ArtifactResolver,
    artifact_ref_from_file,
    file_sha256,
    representation_refs_from_receipt,
)


CHECKSUM = "a" * 64


def _artifact(**changes):
    values = {
        "artifact_id": "artifact-1",
        "artifact_type": ArtifactType.NUMPY_ARRAY,
        "uri": "artifact://test/scene/vector.npy",
        "checksum_sha256": CHECKSUM,
        "size_bytes": 128,
        "format": "npy",
        "producer": "test-producer",
    }
    values.update(changes)
    return ArtifactRef(**values)


def _reference(**changes):
    values = {
        "reference_id": "representation-1",
        "scene_id": "scene-1",
        "representation_type": RepresentationType.OPTICAL_CROMA,
        "modality": "optical",
        "shape": (225, 768),
        "dtype": "float32",
        "spatial_reference": SpatialReference(
            level="token",
            crs="EPSG:32633",
            token_grid=(15, 15),
            mapping_reference="spatial_evidence.scene.token_grid",
            mapping_version="north_up_row_major_120_to_15_v1",
        ),
        "artifact": _artifact(),
        "producer": "CROMAAdapter",
        "provenance_reference": "artifact://test/scene/receipt.json",
        "status": "available",
    }
    values.update(changes)
    return RepresentationRef(**values)


def test_representation_ref_validates_identity_shape_dtype_and_optional_metadata():
    reference = _reference(version="croma-base", preprocessing_version="croma-normalization-v1")
    assert reference.representation_type is RepresentationType.OPTICAL_CROMA
    assert reference.addressable is True
    assert reference.shape == (225, 768)
    assert reference.to_dict()["representation_type"] == "optical_croma"
    assert reference.to_dict()["artifact"]["uri"] == "artifact://test/scene/vector.npy"


@pytest.mark.parametrize("changes", [
    {"reference_id": ""},
    {"scene_id": ""},
    {"representation_type": "invented"},
    {"shape": (225, 0)},
    {"dtype": "float-ish"},
    {"status": "available", "artifact": None},
])
def test_representation_ref_rejects_invalid_contract_values(changes):
    with pytest.raises(ValueError):
        _reference(**changes)


def test_representation_ref_and_spatial_reference_are_immutable():
    reference = _reference()
    with pytest.raises(FrozenInstanceError):
        reference.scene_id = "different"
    with pytest.raises(FrozenInstanceError):
        reference.artifact.checksum_sha256 = "b" * 64
    with pytest.raises(FrozenInstanceError):
        reference.spatial_reference.level = "scene"


def test_artifact_ref_uses_stable_logical_identity_and_validates_checksum():
    artifact = _artifact(checksum_sha256=CHECKSUM.upper())
    assert artifact.checksum_sha256 == CHECKSUM
    assert artifact.to_dict()["artifact_type"] == "numpy_array"
    with pytest.raises(FrozenInstanceError):
        artifact.uri = "artifact://test/scene/other.npy"
    with pytest.raises(ValueError, match="logical reference"):
        _artifact(uri=r"C:\private\vector.npy")
    with pytest.raises(ValueError, match="64 hexadecimal"):
        _artifact(checksum_sha256="not-a-checksum")


def test_representation_set_supports_type_modality_spatial_and_availability_lookup():
    token = _reference()
    scene = _reference(
        reference_id="representation-2",
        shape=(768,),
        spatial_reference=SpatialReference(level="scene"),
        status="not_addressable",
        artifact=None,
    )
    representations = RepresentationSet(references=(token, scene))
    assert representations.find("optical_croma") == (token, scene)
    assert representations.find(RepresentationType.OPTICAL_CROMA, modality="optical", spatial_level="token") == (token,)
    assert representations.find("optical_croma", addressable_only=True) == (token,)
    assert representations.find("sar_croma") == ()


def test_representation_set_registration_returns_a_new_immutable_set():
    empty = RepresentationSet()
    populated = empty.with_reference(_reference())
    assert empty.references == ()
    assert len(populated.references) == 1
    with pytest.raises(ValueError, match="unique"):
        populated.with_reference(_reference())


def test_scene_bundle_round_trip_preserves_representation_and_provenance_references():
    reference = _reference()
    scene = SceneBundle(
        scene_id="scene-1",
        available_modalities=("optical",),
        representations=RepresentationSet(references=(reference,)),
        provenance={"receipt": "artifact://test/scene/receipt.json"},
    )
    payload = scene.to_dict()
    assert payload["representations"]["references"][0]["reference_id"] == reference.reference_id
    assert payload["representations"]["references"][0]["artifact"]["uri"] == reference.artifact.uri
    assert payload["provenance"]["receipt"] == reference.provenance_reference
    json.dumps(payload)


def test_resolver_materializes_verified_numpy_without_paths_in_the_reference(tmp_path):
    root = tmp_path / "store"
    path = root / "scene" / "vector.npy"
    path.parent.mkdir(parents=True)
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(path, array, allow_pickle=False)
    artifact = artifact_ref_from_file(
        path,
        artifact_id="scene:vector",
        artifact_type=ArtifactType.NUMPY_ARRAY,
        logical_uri="artifact://test/scene/vector.npy",
        producer="test",
        format="npy",
    )
    reference = _reference(artifact=artifact, shape=(2, 3), spatial_reference=SpatialReference(level="feature"))
    materialized = ArtifactResolver({"test": root}).load_numpy(reference)
    np.testing.assert_array_equal(materialized, array)
    assert str(tmp_path) not in reference.artifact.uri
    assert artifact.checksum_sha256 == file_sha256(path)
    assert artifact.size_bytes == path.stat().st_size


def test_resolver_fails_closed_for_checksum_mismatch_and_unknown_namespace(tmp_path):
    root = tmp_path / "store"
    path = root / "scene" / "vector.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.ones(2, dtype=np.float32), allow_pickle=False)
    resolver = ArtifactResolver({"test": root})
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolver.resolve(_artifact(size_bytes=path.stat().st_size))
    with pytest.raises(LookupError, match="unconfigured"):
        ArtifactResolver({}).resolve(_artifact())


def test_scientific_receipt_creates_addressable_modality_and_spatial_references(tmp_path):
    root = tmp_path / "audit"
    scene_root = root / "61_39"
    scene_root.mkdir(parents=True)
    arrays = {
        "physical_features": np.arange(62, dtype=np.float32),
        "croma_optical_encodings": np.zeros((225, 768), dtype=np.float32),
    }
    metadata = {}
    for name, array in arrays.items():
        path = scene_root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        metadata[name] = {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype), "sha256": file_sha256(path)}
    receipt = {
        "status": "complete",
        "sample_id": "61_39",
        "arrays": metadata,
        "provenance": {
            "request": {"request_id": "run-61_39"},
            "acquisition": {"crs": "EPSG:32633"},
            "phase1": {"normalization_profile": "croma_readme_patch_8bit_v1"},
            "phase2": {"adapter_version": "1.0"},
        },
    }
    references = representation_refs_from_receipt(receipt, namespace="audit", collection="61_39")
    representations = RepresentationSet(references=references)
    tokens = representations.find("optical_croma", modality="optical", spatial_level="token", addressable_only=True)
    assert len(tokens) == 1
    assert tokens[0].spatial_reference.token_grid == (15, 15)
    assert tokens[0].spatial_reference.mapping_version == "north_up_row_major_120_to_15_v1"
    loaded = ArtifactResolver({"audit": root}).load_numpy(tokens[0])
    assert loaded.shape == (225, 768)


def test_current_compatibility_adapter_exposes_live_materialized_and_not_addressable_states():
    result = {
        "schema_version": "2.0", "analysis_id": "run-1", "timestamp": "now", "status": "completed",
        "query_interpretation": {"task": "joint_optical_sar_analysis", "modalities": ["optical", "sar"]},
        "data_retrieval": {"source": "local", "assets": {"sample_id": "61_39"}},
        "validation": {"checks": {"optical_sar_compatibility": {"crs": True}}},
        "data_cube": {"modalities": ["optical", "sar"], "spatial_dimensions": [120, 120], "metadata": {}},
        "features": {
            "spectral": {"status": "computed", "dimension": 62},
            "deep": {"status": "computed", "source": "official pretrained CROMA base", "pooled_dimension": 2304,
                     "representations": {"optical_encodings": [1, 225, 768], "optical_GAP": [1, 768]}},
            "hybrid": {"status": "computed", "dimension": 192, "source": "existing fusion"},
        },
        "spatial_evidence": {"status": "AVAILABLE", "schema_version": "spatial_evidence_v1",
                             "scene": {"sample_id": "61_39", "crs": "EPSG:32633", "image_shape": [120, 120]},
                             "regions": [{"region_id": "r1"}], "claims": [], "provenance": {}},
        "interpretation": {"status": "ANSWERED"}, "model_results": {"prediction": None},
        "evidence": [], "confidence": {"model_confidence": None}, "preprocessing": [],
        "execution_trace": [], "warnings": [], "error": None, "temporal": {"status": "not requested"},
    }
    adapted = adapt_run_analysis_output(result, {"sample_id": "61_39"})
    references = adapted.scene.representations
    assert references.find("physical_features")[0].status == "materialized"
    assert references.find("croma_scene")[0].shape == (2304,)
    assert references.find("optical_croma", spatial_level="token")[0].status == "not_addressable"
    assert references.find("regions")[0].spatial_reference.level == "region"
    assert references.find("evidence")[0].provenance_reference == "run:run-1"
