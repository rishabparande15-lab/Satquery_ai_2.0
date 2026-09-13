import numpy as np
import pytest
import torch
import json
from pathlib import Path

from src.phase3_6_benchmark import ProbeSplit, assert_no_leakage, fit_probe
from scripts.finalize_phase3_6 import paired_bootstrap


def split(name, ids):
    x=torch.arange(len(ids)*4,dtype=torch.float32).reshape(len(ids),4)
    y=torch.zeros(len(ids),19); y[:,0]=1
    return ProbeSplit(name,x,y,np.asarray(ids))


def test_probe_dimension_and_validation_selection():
    model,report=fit_probe(split("train",["a","b","c"]),split("validation",["d","e"]),max_epochs=2,patience=1,batch_size=2)
    assert model.linear.in_features==4 and report["test_used_for_selection"] is False


def test_leakage_checks_area_and_sentinel_identity():
    samples=[{"patch_id":"a","s1_name":"sa","split":"train"},{"patch_id":"b","s1_name":"sb","split":"validation"},{"patch_id":"c","s1_name":"sc","split":"test"}]
    assert assert_no_leakage(samples,{"train":["a:0"],"validation":["b:0"],"test":["c:0"]})["status"]=="PASS"
    samples[2]["s1_name"]="sa"
    with pytest.raises(ValueError,match="s1_name leakage"): assert_no_leakage(samples,{"train":["a"],"validation":["b"],"test":["c"]})


def test_shared_prediction_comparison_rejects_target_identity_mismatch(tmp_path):
    common={"predictions":torch.tensor([[1.0]+[0.0]*18]),"truth":torch.tensor([[1.0]+[0.0]*18]),"area_ids":["area"]}
    torch.save(common,tmp_path/"r.pt"); bad={**common,"area_ids":["other"]}; torch.save(bad,tmp_path/"s.pt")
    with pytest.raises(ValueError,match="align"): paired_bootstrap(tmp_path/"r.pt",tmp_path/"s.pt",repeats=2)


def test_completed_benchmark_manifests_metrics_annotations_and_reproducibility():
    root=Path("experiments/comparison")
    required=[root/"dataset/dataset_manifest.json",root/"dataset/shared_evaluation_manifest.json",root/"ravi_baseline/metrics.json",root/"satquery_hybrid/metrics.json",root/"comparison/comparison_report.json"]
    if not all(path.exists() for path in required): pytest.skip("Generated Phase 3.6 benchmark outputs are not present")
    dataset=json.loads(required[0].read_text()); shared=json.loads(required[1].read_text()); ravi=json.loads(required[2].read_text()); sat=json.loads(required[3].read_text()); comparison=json.loads(required[4].read_text())
    assert dataset["sample_count"]==1000 and dataset["split_counts"]=={"train":600,"validation":200,"test":200}
    assert len({x["sample_id"] for x in dataset["samples"]})==1000 and len({x["sentinel1_identity"] for x in dataset["samples"]})==1000
    assert len(shared["test_area_ids"])==200 and shared["eligible_block_count"]==44723
    assert ravi["metrics"]["dominant_class_accuracy"]==pytest.approx(.538742985409652) and ravi["metrics"]["coverage_mae_pp"]==pytest.approx(6.1552576860794765)
    assert sat["reproducibility"]["status"]=="EXACTLY REPRODUCIBLE" and comparison["leakage"]["status"]=="PASS"
    assert comparison["bigearthnet_txt"]["matched_images"]==955 and comparison["bigearthnet_txt"]["total_records"]==20453
