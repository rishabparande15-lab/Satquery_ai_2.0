from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "experiments" / "pipeline3_5000"
OUT = PIPELINE / "s2_acquisition"
EXPECTED_BANDS = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_missing_before_freezes_exact_original_4000_train_ids():
    before = load(OUT / "missing_before.json")
    assert before["missing_count"] == 4000
    assert len(before["area_ids"]) == len(set(before["area_ids"])) == 4000
    assert {record["area_id"] for record in before["records"]} == set(before["area_ids"])
    assert {record["split"] for record in before["records"]} == {"train"}
    assert all(set(record["required_bands"]) == EXPECTED_BANDS for record in before["records"])


def test_matched_s2_covers_the_exact_selected_population_with_checksums():
    manifest_ids = {row["area_id"] for row in load(PIPELINE / "dataset_manifest.json")["rows"]}
    matched = load(OUT / "matched_s2.json")
    records = matched["records"]
    assert matched["matched_count"] == 5000
    assert {record["area_id"] for record in records} == manifest_ids
    assert all(record["band_count"] == 12 for record in records)
    assert all(set(record["band_sha256"]) == EXPECTED_BANDS for record in records)
    assert all(set(record["band_size_bytes"]) == EXPECTED_BANDS for record in records)
    assert all(record["validation_status"] == "valid" for record in records)
    assert all(record["finite"] and record["north_up"] and record["mask_matches_nodata"] for record in records)
    acquired = [record for record in records if record["acquisition_source_record_sha256"] is not None]
    assert len(acquired) == 4000
    assert {record["acquisition_source_revision"] for record in acquired} == {
        "118d1b6285c080ba8e4078414e1b8a243b18c9bd"
    }


def test_unmatched_candidates_and_duplicates_are_explicit_and_non_leaking():
    manifest_ids = {row["area_id"] for row in load(PIPELINE / "dataset_manifest.json")["rows"]}
    unmatched = load(OUT / "unmatched_candidates.json")
    assert unmatched["unmatched_area_count"] == 3
    assert all(record["area_id"] not in manifest_ids for record in unmatched["records"])
    assert all(record["reason"] == "area_id_not_in_exact_selected_5000_manifest" for record in unmatched["records"])
    duplicates = load(OUT / "duplicates.json")
    assert duplicates["duplicate_area_band_groups"] == []
    assert duplicates["duplicate_whole_area_content_groups"] == []
    assert duplicates["cross_area_band_content_groups"] == []
    assert duplicates["cross_split_duplicate_content_groups"] == []
    assert len(duplicates["same_area_cross_band_content_groups"]) == 7


def test_final_acquisition_status_is_complete_and_split_preserving():
    status = load(OUT / "acquisition_status.json")
    assert status["selected_count"] == 5000
    assert status["preexisting_valid_s2_count"] == 1000
    assert status["recovered_s2_count"] == 4000
    assert status["rejected_candidate_area_count"] == 3
    assert status["matched_valid_s2_count"] == 5000
    assert status["unresolved_selected_count"] == 0
    assert status["split_counts"] == {"train": 4600, "validation": 200, "test": 200}
    assert status["completeness_gate_ready"] is True
