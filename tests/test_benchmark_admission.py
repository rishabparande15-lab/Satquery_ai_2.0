import pytest

from src.benchmark_admission import BenchmarkAdmissionError, normalize_development_record


def record(**extra):
    return {"benchmark": "VRSBench", "record_id": "v-1", "image_path": "image.png", "question": "What is visible?",
            "reference_answer": "road", "task": "VQA", "split": "validation", "source": "author release", "provenance": {"revision": "r1"}, **extra}


def test_normalizes_explicit_validation_without_model_execution():
    value = normalize_development_record(record())
    assert value.split == "validation" and value.receipt()["record_id"] == "v-1"


@pytest.mark.parametrize("split", ["test", "", "unknown"])
def test_rejects_test_or_undefined_split(split):
    with pytest.raises(BenchmarkAdmissionError):
        normalize_development_record(record(split=split))
