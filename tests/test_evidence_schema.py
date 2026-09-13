import numpy as np
import pytest
from src.evidence_schema import EvidenceQueries, build_spatial_evidence, canonical_sha256, evidence_strength, group_regions, token_extent, validate_evidence

TRANSFORM=[10,0,1000,0,-10,2200,0,0,1]

def inputs(optical=True,sar=True,croma=True):
    o=np.ones((12,120,120),dtype=np.float32) if optical else None
    if o is not None:
        o[7]=3; o[3]=1; o[2]=1; o[10]=2; o[1]=1
        o[7,:60,:60]=8
    s=np.stack((np.full((120,120),-8),np.full((120,120),-14))).astype(np.float32) if sar else None
    c={m:{"path":f"{m}.npy","sha256":"a"*64,"shape":[225,768]} for m in ("optical","sar","joint")} if croma else None
    return dict(sample_id="x",optical=o,sar=s,transform=TRANSFORM,crs="EPSG:32633",scene_metadata={},
                provenance={"pixel_source_artifact":"probe/receipt.json","source_hashes":{"x":"b"*64}},croma=c,predictions=None)

def test_token_mapping_and_coordinates():
    assert token_extent(0,TRANSFORM)["map_bounds"] == [1000,2120,1080,2200]
    assert token_extent(224,TRANSFORM)["pixel_bounds"] == [112,112,120,120]

def test_region_grouping_is_four_connected_and_deterministic():
    assert group_regions([16,0,1,15,224]) == [[0,1,15,16],[224]]

def test_strength_is_not_confidence_and_handles_missing():
    assert evidence_strength(coverage=.4,support_fraction=1,largest_region_fraction=1)=="UNAVAILABLE"
    assert evidence_strength(coverage=1,support_fraction=.4,largest_region_fraction=.3)=="STRONG"

def test_physical_aggregation_claim_traceability_and_scene():
    e=build_spatial_evidence(**inputs()); validate_evidence(e)
    assert len(e["tokens"])==225 and e["tokens"][0]["features"]["NDVI"]["valid_pixel_count"]==64
    assert e["claims"] and all(c["support"][0]["source_artifact"] for c in e["claims"])
    assert e["scene"]["summary"]["valid_data_coverage"]==1

def test_optical_only_missing_sar_and_joint():
    e=build_spatial_evidence(**inputs(sar=False,croma=False))
    assert e["sensor_views"]["optical"]["status"]=="AVAILABLE"
    assert e["sensor_views"]["sar"]["status"]=="UNAVAILABLE" and e["sensor_views"]["joint"]["status"]=="UNAVAILABLE"

def test_sar_only_missing_optical():
    e=build_spatial_evidence(**inputs(optical=False,croma=False))
    assert e["sensor_views"]["sar"]["status"]=="AVAILABLE"
    assert "NDVI" not in e["tokens"][0]["features"]

def test_joint_is_latent_not_semantic_claim():
    e=build_spatial_evidence(**inputs())
    assert e["tokens"][0]["croma"]["joint"]["semantic_status"].startswith("LATENT")
    assert all(c["sensor_view"] != "JOINT" for c in e["claims"])

def test_prediction_unavailable_is_explicit():
    e=build_spatial_evidence(**inputs())
    assert e["tokens"][0]["prediction_status"]=="UNAVAILABLE"
    assert e["scene"]["summary"]["prediction_summary"]["status"]=="UNAVAILABLE"

def test_supplied_predictions_are_typed_attached_and_aggregated():
    kw=inputs(); kw["predictions"]={0:[{"class":"water","score":.8,"model":"test","model_version":"v1","evidence_type":"MODEL_PREDICTED"}],1:[{"class":"water","score":.6,"model":"test","model_version":"v1","evidence_type":"MODEL_PREDICTED"}]}
    e=build_spatial_evidence(**kw); q=EvidenceQueries(e)
    assert e["tokens"][0]["prediction_status"]=="MODEL_PREDICTED"
    assert e["scene"]["summary"]["prediction_summary"]["dominant_class"]=="water"
    assert q.get_class_regions("water")[0]["token_indices"]==[0,1]

def test_queries_and_missing_feature():
    q=EvidenceQueries(build_spatial_evidence(**inputs()))
    assert q.get_token_evidence(0)["token_index"]==0
    assert q.get_feature_map_evidence("missing")=={"status":"UNAVAILABLE"}
    assert q.get_sensor_evidence("optical")["status"]=="AVAILABLE" and q.get_supported_claims()

def test_provenance_survives():
    e=build_spatial_evidence(**inputs())
    assert e["provenance"]["source_hashes"]["x"]=="b"*64
    assert e["provenance"]["spatial_mapping_version"]

def test_unsupported_claim_rejected():
    e=build_spatial_evidence(**inputs()); e["claims"][0]["support"]=[]
    with pytest.raises(ValueError,match="unsupported claim"): validate_evidence(e)

def test_deterministic_reproduction():
    assert canonical_sha256(build_spatial_evidence(**inputs())) == canonical_sha256(build_spatial_evidence(**inputs()))

def test_neither_sensor_is_rejected():
    with pytest.raises(ValueError): build_spatial_evidence(**inputs(optical=False,sar=False,croma=False))
