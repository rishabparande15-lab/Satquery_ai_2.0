from scripts.phase3p3_sar_native_supervision import role
import json
from pathlib import Path
import json
from pathlib import Path

def test_water_binary_label_filter_rejects_water_adjacent_ambiguity():
    assert role(['Inland waters']) == 'yes'
    assert role(['Arable land']) == 'no'
    assert role(['Inland wetlands']) is None

def test_water_binary_label_filter_is_not_optical_metadata_task():
    assert role(['Marine waters', 'Urban fabric']) == 'yes'


def test_completed_pool_is_balanced_split_safe_and_provenanced():
    root = Path(__file__).resolve().parents[1] / 'artifacts/training/phase3p/phase3p3_sar_native_supervision'
    summary = json.loads((root / 'phase3p3_summary.json').read_text())
    distribution = json.loads((root / 'phase3p3_distribution_audit.json').read_text())
    train = json.loads((root / 'phase3p3_train_manifest.json').read_text())['records']
    valid = json.loads((root / 'phase3p3_validation_manifest.json').read_text())['records']
    assert summary['test_access_count'] == 0 and len(train) == 80 and len(valid) == 20
    assert not ({r['sentinel2_patch_id'] for r in train} & {r['sentinel2_patch_id'] for r in valid})
    assert distribution['binary_answer_distribution']['all'] == {'no': 50, 'yes': 50}
    assert all(r['correct_vs_shuffled_target_changes'] for r in train + valid)


def test_completed_pool_is_balanced_split_safe_and_provenanced():
    root = Path(__file__).resolve().parents[1] / 'artifacts/training/phase3p/phase3p3_sar_native_supervision'
    summary = json.loads((root / 'phase3p3_summary.json').read_text())
    distribution = json.loads((root / 'phase3p3_distribution_audit.json').read_text())
    train = json.loads((root / 'phase3p3_train_manifest.json').read_text())['records']
    valid = json.loads((root / 'phase3p3_validation_manifest.json').read_text())['records']
    assert summary['test_access_count'] == 0 and len(train) == 80 and len(valid) == 20
    assert not ({r['sentinel2_patch_id'] for r in train} & {r['sentinel2_patch_id'] for r in valid})
    assert distribution['binary_answer_distribution']['all'] == {'no': 50, 'yes': 50}
    assert all(r['correct_vs_shuffled_target_changes'] for r in train + valid)
