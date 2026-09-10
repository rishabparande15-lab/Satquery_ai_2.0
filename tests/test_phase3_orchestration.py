from datetime import datetime, timezone
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from src.phase1_foundation import OPTICAL_BANDS,SAR_BANDS
from src.phase3_orchestration import *

def aoi(): return AOI.from_input({"type":"bbox","bounds":[0,0,.01,.01]})
def temporal(): return TemporalRequest.create(start="2026-01-01",end="2026-01-31")
def scene(id="x",source="sentinel-2",modality=Modality.OPTICAL,bands=OPTICAL_BANDS,date=10,cloud=1,native=10):
    return SceneCandidate(id,source,"Sentinel-2" if source=="sentinel-2" else "Sentinel-1","MSI" if source=="sentinel-2" else "C-SAR",modality,datetime(2026,1,date,tzinfo=timezone.utc),native,"EPSG:4326",(0,0,.02,.02),1,tuple(bands),cloud)

def test_aoi_contracts_and_invalid_geometry():
    assert AOI.from_input({"type":"point","latitude":12,"longitude":77}).area_km2==0
    assert aoi().geometry["type"]=="Polygon"
    assert AOI.from_input({"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}).area_km2>0
    assert AOI.from_input({"type":"Feature","geometry":{"type":"Point","coordinates":[77,12]}}).crs=="EPSG:4326"
    with pytest.raises(OrchestrationError) as e: AOI.from_input({"type":"point","latitude":91,"longitude":0})
    assert e.value.code==FailureCode.INVALID_AOI
    with pytest.raises(OrchestrationError): AOI.from_input("Bengaluru")

def test_temporal_contract_variants():
    assert TemporalRequest.create(acquisition_date="2026-01-01").start==date(2026,1,1)
    assert temporal().end==date(2026,1,31)
    t=TemporalRequest.create(before="2026-01-01",after="2026-02-01",tolerance_days=2); assert t.before<t.after
    with pytest.raises(OrchestrationError): TemporalRequest.create(start="2026-02-01",end="2026-01-01")

def test_registry_modality_and_resolution_truth():
    assert get_source("sentinel-1").bands==SAR_BANDS and get_source("sentinel-2").bands==OPTICAL_BANDS
    with pytest.raises(OrchestrationError): get_source("risat")
    r=ResolutionContract.evaluate(5,10); assert r.resolution_status=="unavailable_at_requested_resolution" and r.processing_gsd==10
    up=ResolutionContract.evaluate(5,10,processing=5); assert up.information_status=="resampled_from_10m"

def test_availability_filter_selection_and_no_scene():
    req=AvailabilityRequest(aoi(),temporal(),Modality.OPTICAL,10,("sentinel-2",),5)
    chosen,reason=select_scene(req,[scene("late",date=20,cloud=2),scene("near",date=16,cloud=3)])
    assert chosen.scene_id=="near" and "deterministic" in reason
    assert LocalSceneAdapter([]).search(req).error_code==FailureCode.NO_SCENE_AVAILABLE
    with pytest.raises(OrchestrationError) as e: select_scene(req,[scene("cloudy",cloud=90)])
    assert e.value.code==FailureCode.NO_SCENE_AVAILABLE

def write_raster(path,bands,*,crs="EPSG:4326",transform=None,data=None):
    data=np.ones((len(bands),120,120),dtype=np.float32) if data is None else data
    with rasterio.open(path,"w",driver="GTiff",height=120,width=120,count=len(bands),dtype="float32",crs=crs,transform=transform or from_origin(0,.02,.02/120,.02/120)) as ds:
        ds.write(data)
        for i,b in enumerate(bands,1): ds.set_band_description(i,b)

def test_raster_validation_band_order_finiteness_and_coverage(tmp_path):
    p=tmp_path/"o.tif"; write_raster(p,OPTICAL_BANDS); s=scene(); meta=validate_raster(p,s,aoi(),Modality.OPTICAL); assert meta["array"].shape==(12,120,120)
    bad=tmp_path/"bad.tif"; write_raster(bad,tuple(reversed(OPTICAL_BANDS)))
    with pytest.raises(OrchestrationError) as e: validate_raster(bad,s,aoi(),Modality.OPTICAL)
    assert e.value.code==FailureCode.INVALID_RASTER
    nan=tmp_path/"nan.tif"; values=np.ones((12,120,120),dtype=np.float32); values[0,0,0]=np.nan; write_raster(nan,OPTICAL_BANDS,data=values)
    with pytest.raises(OrchestrationError): validate_raster(nan,s,aoi(),Modality.OPTICAL)
    outside=AOI.from_input({"type":"bbox","bounds":[1,1,2,2]})
    with pytest.raises(OrchestrationError) as e: validate_raster(p,s,outside,Modality.OPTICAL)
    assert e.value.code==FailureCode.INSUFFICIENT_AOI_COVERAGE

def test_spatial_pair_mismatch():
    base={"crs":"EPSG:4326","bounds":(0,0,1,1),"resolution":(10,10),"transform":(10,0,0,0,-10,1,0,0,1),"array":np.ones((12,120,120))}
    other={**base,"array":np.ones((2,120,120)),"bounds":(1,1,2,2)}
    with pytest.raises(OrchestrationError) as e: validate_spatial_pair(base,other)
    assert e.value.code==FailureCode.MODALITY_PAIR_INCOMPATIBLE

class FakeCroma(torch.nn.Module):
    def forward(self,optical_images,SAR_images):
        n=len(optical_images); z=torch.zeros(n,225,768); g=torch.zeros(n,768)
        return {"optical_encodings":z,"SAR_encodings":z,"joint_encodings":z,"optical_GAP":g,"SAR_GAP":g,"joint_GAP":g}
class Physical:
    def extract(self,o,s,ob,sb): return np.array([1,2],dtype=np.float32),{"dimension":2,"feature_names":["a","b"],"schema":"test"}

def test_phase3_bridge_preserves_identity_and_provenance():
    ds=ValidatedSatelliteDataset("req",("s2","s1"),np.ones((12,120,120),np.float32),np.ones((2,120,120),np.float32),aoi(),"EPSG:32632",(0,0,1200,1200),10,10,{"optical":OPTICAL_BANDS,"sar":SAR_BANDS},("2026-01-01","2026-01-01"),{"o":"abc"},{"provider":"local"})
    out=bridge_to_phase2(ds,FakeCroma(),Physical(),croma_provenance={"checkpoint":"x","checkpoint_sha256":"abc"})
    assert out.identities[0].patch_id=="req" and out.pooled_croma_features.shape==(1,2304)
    assert out.provenance[0]["source_hashes"]=={"o":"abc"}
