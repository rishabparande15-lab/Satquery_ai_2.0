from pathlib import Path

import pytest

from src import gee_experiment_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_mode_is_explicit_and_does_not_claim_gee(monkeypatch):
    status = {"package_available": True, "authenticated": True, "error": None}
    monkeypatch.setattr(gee_experiment_runtime, "gee_status", lambda: status)

    mode, returned = gee_experiment_runtime.select_mode("local")

    assert mode == "local_reference"
    assert returned is status


def test_required_gee_mode_fails_closed_when_authentication_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        gee_experiment_runtime,
        "gee_status",
        lambda: {
            "package_available": True,
            "authenticated": False,
            "error": "GEE initialization failed: unavailable",
        },
    )

    with pytest.raises(RuntimeError, match="GEE initialization failed"):
        gee_experiment_runtime.select_mode("gee")


def test_pipeline3_experiment_has_no_pipeline2_import_dependency():
    pipeline3 = PROJECT_ROOT / "experiments" / "pipelines" / "gee_croma"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in pipeline3.glob("*.py"))

    assert "experiments.pipelines.gee" not in sources


def test_production_source_does_not_import_experiment_packages():
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "src").glob("*.py"))

    assert "experiments.pipelines" not in sources
