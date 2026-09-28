"""Independent specifications; metadata is conservative and unknowns remain unknown."""
from .contracts import BenchmarkSpec, BenchmarkStatus, UNKNOWN

_PROVENANCE = ("benchmark_revision", "dataset_revision", "split", "sample_manifest", "model_revision", "adapter_revision", "preprocessing_version", "evaluator_version", "random_seed", "configuration", "prediction_artifact_hash", "result_artifact_hash")
_LEAKAGE = ("no_train_test_overlap", "no_duplicate_sample_ids", "known_split", "valid_target_linkage", "benchmark_revision_match")
_LICENSE = ("image_license", "annotation_license", "derivative_use_terms")

BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {
    "RSVQA_BINARY": BenchmarkSpec("RSVQA_BINARY", "single_image_vqa_binary", "optical_rgb", expected_input={"image_count": 1}, expected_target={"answer_type": "binary"}, evaluator_id="RSVQA_OFFICIAL_EVALUATOR_UNKNOWN", metrics=("accuracy",), required_provenance=_PROVENANCE, license_requirements=_LICENSE, leakage_requirements=_LEAKAGE, expected_output_format={"answer": "string"}),
    "RSVQA_MULTIPLE_CHOICE": BenchmarkSpec("RSVQA_MULTIPLE_CHOICE", "single_image_vqa_multiple_choice", "optical_rgb", expected_input={"image_count": 1, "choices": "required"}, expected_target={"answer_type": "choice"}, evaluator_id="RSVQA_OFFICIAL_EVALUATOR_UNKNOWN", metrics=("accuracy",), required_provenance=_PROVENANCE, license_requirements=_LICENSE, leakage_requirements=_LEAKAGE, expected_output_format={"answer": "choice_key"}),
    "RSVQA_OPEN_ENDED": BenchmarkSpec("RSVQA_OPEN_ENDED", "single_image_vqa_open_ended", "UNKNOWN", expected_input={"image_count": 1}, expected_target={"answer_type": "UNKNOWN"}, evaluator_id=UNKNOWN, metrics=(), required_provenance=_PROVENANCE, license_requirements=_LICENSE, leakage_requirements=_LEAKAGE, expected_output_format={"answer": "string"}),
    "VRSBENCH_GROUNDING": BenchmarkSpec("VRSBENCH_GROUNDING", "visual_grounding", "optical_rgb", expected_input={"image_count": 1, "query": "required"}, expected_target={"geometry": "bbox", "coordinate_semantics": "required"}, evaluator_id="VRSBENCH_OFFICIAL_EVALUATOR_UNKNOWN", metrics=(), required_provenance=_PROVENANCE + ("coordinate_space", "image_dimensions"), license_requirements=_LICENSE, leakage_requirements=_LEAKAGE, expected_output_format={"geometry": "bbox"}),
    "CDVQA_CHANGE_VQA": BenchmarkSpec("CDVQA_CHANGE_VQA", "temporal_change_vqa", "temporal_UNKNOWN", expected_input={"image_count": 2, "temporal_order": "required"}, expected_target={"answer_type": "UNKNOWN"}, evaluator_id="CDVQA_OFFICIAL_EVALUATOR_UNKNOWN", metrics=(), required_provenance=_PROVENANCE + ("pair_id", "t1_id", "t2_id", "temporal_order", "registration_status"), license_requirements=_LICENSE, leakage_requirements=_LEAKAGE + ("no_pair_overlap",), expected_output_format={"answer": "string"}, dataset_status=BenchmarkStatus.DATASET_UNAVAILABLE),
    "ISRO_SAC_HIDDEN_FINAL": BenchmarkSpec("ISRO_SAC_HIDDEN_FINAL", "hidden_final_evaluation", "UNKNOWN", expected_input={"contract": "UNKNOWN"}, expected_target={"hidden": True}, evaluator_id=UNKNOWN, metrics=(), required_provenance=_PROVENANCE, license_requirements=("access_authorization", "evaluation_terms"), leakage_requirements=_LEAKAGE, expected_output_format={"submission": "UNKNOWN"}, dataset_status=BenchmarkStatus.DATASET_UNVERIFIED),
}


def get_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    try:
        return BENCHMARK_REGISTRY[benchmark_id]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark: {benchmark_id}") from exc
