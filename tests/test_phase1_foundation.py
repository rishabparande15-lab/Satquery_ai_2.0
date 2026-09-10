import json
import numpy as np
import pytest
import torch

from src.phase1_foundation import (CLASSES, FrozenCromaFoundation, ReferenceCoverageHead,
    SampleIdentity, coverage_metrics, link_annotations, normalize_patch, patch_label_targets,
    prepare_batch, validate_split_integrity, write_experiment_report)


def identity(name="p1", split="train"):
    return SampleIdentity(name, f"s1_{name}", f"s2_{name}", split, {"date":"2026-01-01"}, "EPSG:32632", (0,0,1200,1200), (10,10), {"source":"fixture"})


def test_data_contract_and_deterministic_reference_normalization():
    optical=torch.arange(12*120*120, dtype=torch.float32).reshape(1,12,120,120)
    sar=torch.arange(2*120*120, dtype=torch.float32).reshape(1,2,120,120)
    first=prepare_batch(optical,sar,[identity()]); second=prepare_batch(optical,sar,[identity()])
    assert first.optical.shape==(1,12,120,120) and first.sar.shape==(1,2,120,120)
    assert torch.equal(first.optical,second.optical) and first.normalization["reference_grid"]=="B02"
    constant, meta=normalize_patch(torch.ones(1,120,120))
    assert constant.unique().item()==pytest.approx(127/255) and meta["constant_channels"]==[0]


def test_bad_contract_and_metadata_fail_clearly():
    with pytest.raises(ValueError, match="Optical input"):
        prepare_batch(torch.zeros(1,11,120,120),torch.zeros(1,2,120,120),[identity()])
    with pytest.raises(ValueError, match="split"):
        prepare_batch(torch.zeros(1,12,120,120),torch.zeros(1,2,120,120),[identity(split="holdout")])


class FakeCroma(torch.nn.Module):
    def forward(self, optical_images, SAR_images):
        n=len(optical_images); z=torch.zeros(n,225,768)
        return {"optical_encodings":z,"SAR_encodings":z,"joint_encodings":z,
                "optical_GAP":z.mean(1),"SAR_GAP":z.mean(1),"joint_GAP":z.mean(1)}


def test_frozen_croma_six_output_contract():
    batch=prepare_batch(torch.zeros(1,12,120,120),torch.zeros(1,2,120,120),[identity()])
    wrapper=FrozenCromaFoundation(FakeCroma(),{"checkpoint":"x.pt","checkpoint_sha256":"abc"})
    output=wrapper(batch)
    assert output["joint_encodings"].shape==(1,225,768) and output["joint_GAP"].shape==(1,768)


def test_official_targets_preserve_unknown_and_row_major_order():
    maps=torch.full((1,120,120),111,dtype=torch.int64); maps[:,0:8,0:4]=0
    target=patch_label_targets(maps)
    assert len(CLASSES)==19 and target["class_fractions"].shape==(1,225,19)
    assert target["class_fractions"][0,0,0]==.5 and target["unlabeled_fraction"][0,0]==.5
    assert target["class_fractions"][0,1,0]==1 and not target["fully_labeled_mask"][0,0]


def test_reference_baseline_is_fraction_not_confidence():
    head=ReferenceCoverageHead(); fractions=head.fractions(torch.zeros(2,225,768))
    assert fractions.shape==(2,225,19) and torch.allclose(fractions.sum(-1),torch.ones(2,225))


def test_split_leakage_rejected():
    validate_split_integrity({"train":["a"],"validation":["b"],"test":["c"]})
    with pytest.raises(ValueError,match="leakage"):
        validate_split_integrity({"train":["a"],"validation":["a"],"test":["c"]})


def test_metrics_provenance_and_annotation_linkage(tmp_path):
    y=np.eye(19)[[0,1]]; report=coverage_metrics(y,y,["a","b"],bootstrap_repeats=5)
    assert report["mae_pp"]==0 and report["dominant_class_accuracy"]==1
    path=write_experiment_report(tmp_path/"report.json",metrics=report,configuration={"seed":1},identities=[identity()],model={"kind":"REFERENCE BASELINE"},feature_contract={"shape":[1,225,768]})
    assert json.loads(path.read_text())["status"]=="complete" and path.with_suffix(".json.receipt.json").exists()
    rows=link_annotations([identity()], [{"ID":"q1","patch_id":"s2_p1","s1_name":"s1_p1","type":"binary","input":"water?","output":"yes","split":"train"}])
    assert rows[0]["training_eligible"] and not rows[0]["semantics_validated"] and not rows[0]["feature_token_labels"]
