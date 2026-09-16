from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_pipeline3_5000 import CANONICAL_OUT, OUT as AUDIT_OUTPUT, validate_output_root


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "pipeline3_5000"


def _load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_exact_5000_manifest_has_one_explicit_row_per_selected_area():
    manifest = _load("dataset_manifest.json")
    rows = manifest["rows"]
    assert len(rows) == 5000
    assert len({row["area_id"] for row in rows}) == 5000
    assert Counter(row["split"] for row in rows) == {"train": 4600, "validation": 200, "test": 200}
    assert all(row["availability"]["s1_vv"] and row["availability"]["s1_vh"] for row in rows)
    assert all(row["availability"]["reference"] and row["availability"]["metadata"] for row in rows)
    assert sum(row["validation"]["complete_multimodal"] for row in rows) == 5000
    assert sum("matching_s2_12_band_rasters_unavailable" in row["validation"]["exclusion_reasons"] for row in rows) == 0


def test_exact_split_is_deterministic_disjoint_and_matches_embedded_counts():
    split = _load("split_manifest.json")
    assert split["counts"] == {"train": 4600, "validation": 200, "test": 200}
    assert split["declared_counts_match_metadata"] is True
    assert split["baseline_ids_are_subsets"] is True
    assert split["baseline_id_counts"] == {"train": 600, "validation": 200, "test": 200}
    assert split["additional_id_counts"] == {"train": 4000, "validation": 0, "test": 0}
    assert all(not values for values in split["intersections"].values())
    assert split["split_sha256"] == "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"


def test_completeness_gate_record_is_preserved_and_later_training_is_receipted():
    audit = _load("audit_summary.json")
    assert audit["status"] == "ready_for_training"
    assert audit["decision"] is None
    assert audit["completeness_gate_passed"] is True
    assert audit["training_started"] is False
    assert audit["availability"]["complete_optical_areas"] == 5000
    assert audit["availability"]["complete_sar_areas"] == 5000
    assert audit["availability"]["complete_reference_areas"] == 5000
    assert audit["availability"]["complete_multimodal_by_split"] == {"train": 4600, "validation": 200, "test": 200}
    baseline = _load("baseline_metrics.json")
    final = _load("final_metrics.json")
    receipt = _load("final/test_receipt.json")
    assert baseline["status"] == "complete_validation_and_final_test"
    assert final["status"] == "complete"
    assert baseline["areas_used"] == final["areas_used"] == 5000
    assert final["selected_model"] == "hybrid"
    assert set(final["ablations"]) == {"constant", "physical", "optical_croma", "sar_croma", "joint_croma", "hybrid"}
    assert receipt["status"] == "consumed_once"
    assert receipt["test_area_count"] == 200
    assert receipt["scientific_fingerprint"] == _load("final/fingerprint.json")["fingerprint"]


def test_audit_fingerprints_match_machine_readable_artifacts():
    reproducibility = _load("reproducibility.json")
    fingerprint = _load("fingerprint.json")
    assert reproducibility["dataset_manifest_sha256"] == _sha256(OUT / "dataset_manifest.json")
    assert reproducibility["split_manifest_sha256"] == _sha256(OUT / "split_manifest.json")
    assert reproducibility["audit_summary_sha256"] == _sha256(OUT / "audit_summary.json")
    canonical = json.dumps(fingerprint["basis"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert fingerprint["fingerprint"] == hashlib.sha256(canonical).hexdigest()
    assert fingerprint["status"].startswith("dataset_gate_fingerprint_only")


def test_canonical_scientific_metrics_and_fingerprints_are_protected():
    baseline = _load("baseline_metrics.json")
    final = _load("final_metrics.json")
    final_config = _load("final/config.json")
    final_fingerprint = _load("final/fingerprint.json")

    assert baseline["status"] == "complete_validation_and_final_test"
    assert final["status"] == "complete"
    assert baseline["test"] == final["test"]
    assert baseline["test"]["mae_pp"] == pytest.approx(4.103428673440559, abs=1e-12)
    assert baseline["test"]["rmse_pp"] == pytest.approx(9.587331212294444, abs=1e-12)
    assert baseline["test"]["dominant_class_accuracy"] == pytest.approx(0.65, abs=1e-12)
    assert final_config["dataset_fingerprint"] == "7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625"
    assert final_config["split_fingerprint"] == "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"
    assert final_config["model_state_sha256"] == "85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634"
    assert final_fingerprint["fingerprint"] == "ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae"


def test_dataset_audit_runtime_output_cannot_target_canonical_records():
    assert AUDIT_OUTPUT.resolve() != CANONICAL_OUT.resolve()
    with pytest.raises(RuntimeError, match="cannot write canonical"):
        validate_output_root(CANONICAL_OUT)


def test_leakage_findings_distinguish_within_train_repeats_from_cross_split_leakage():
    leakage = _load("audit_summary.json")["leakage"]
    assert leakage["area_id_duplicates"] == []
    assert leakage["s1_identity_duplicates"] == []
    assert leakage["duplicate_s1_content_groups"] == []
    assert len(leakage["duplicate_s2_content_groups"]) == 7
    assert leakage["duplicate_s2_cross_area_groups"] == []
    assert leakage["duplicate_s2_cross_split_groups"] == []
    assert len(leakage["duplicate_reference_content_groups"]) == 103
    assert leakage["duplicate_reference_cross_split_groups"] == []
    assert leakage["positive_spatial_overlap_count"] == 138
    assert leakage["cross_split_positive_spatial_overlap_count"] == 0
    assert leakage["classification"]["unintentional_leakage_found"] is False
