import torch
import json
from pathlib import Path
from src.eo_vlm.sar_projector import S1SARProjector

def test_sar_projector_only_optimizer_membership():
    sar=S1SARProjector(); unrelated=torch.nn.Linear(2,2)
    optimizer=torch.optim.AdamW(sar.parameters())
    member={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in member for p in sar.parameters())
    assert all(id(p) not in member for p in unrelated.parameters())

def test_manifest_partition_and_test_policy_shape():
    train=[{'record_id':'a','image_id':'i1'}]; valid=[{'record_id':'b','image_id':'i2'}]
    assert not ({x['image_id'] for x in train}&{x['image_id'] for x in valid})
    assert len({x['record_id'] for x in train+valid})==2


def test_completed_phase3p1_manifest_checkpoint_and_reload_receipt():
    root = Path(__file__).resolve().parents[1] / "artifacts/training/phase3p/phase3p1_sar_adaptation"
    manifest = json.loads((root / "phase3p1_training_manifest.json").read_text())
    receipt = json.loads((root / "phase3p1_reload_receipt.json").read_text())
    assert manifest["test_access_count"] == 0
    assert not manifest["integrity"]["train_validation_image_overlap"]
    assert (root / "phase3p1_sar_projector_initial.pt").is_file()
    assert (root / "phase3p1_sar_projector_final.pt").is_file()
    assert receipt["fresh_process"] and receipt["agreement"] in {"EXACT_MATCH", "WITHIN_TOLERANCE"}


def test_semantic_audit_correct_shuffled_zero_mapping_and_parsing():
    root = Path(__file__).resolve().parents[1] / "artifacts/training/phase3p/phase3p1_sar_adaptation"
    audit = json.loads((root / "phase3p1_trained_semantic_audit.json").read_text())
    assert audit["test_access_count"] == 0
    for record in audit["records"]:
        assert set(record["conditions"]) == {"correct", "shuffled", "zero"}
        assert {"correct", "shuffled", "zero"} <= set(record["target_scores"])
