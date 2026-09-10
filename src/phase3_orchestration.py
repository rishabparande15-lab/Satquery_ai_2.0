"""Typed, source-independent operational satellite-data backbone for Phase 3."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence
from uuid import uuid4

import numpy as np
import rasterio
import torch
from rasterio.warp import transform_bounds

from .phase1_foundation import NORMALIZATION_PROFILE, OPTICAL_BANDS, SAR_BANDS, SampleIdentity, prepare_batch
from .phase2_integration import Phase2PipelineInput, adapt_phase1_output

PHASE3_VERSION = "1.0"
PHASE2_ADAPTER_VERSION = "1.0"


class FailureCode(str, Enum):
    INVALID_AOI = "INVALID_AOI"
    INVALID_TEMPORAL_REQUEST = "INVALID_TEMPORAL_REQUEST"
    UNSUPPORTED_SOURCE = "UNSUPPORTED_SOURCE"
    NO_SCENE_AVAILABLE = "NO_SCENE_AVAILABLE"
    REQUESTED_RESOLUTION_UNAVAILABLE = "REQUESTED_RESOLUTION_UNAVAILABLE"
    INSUFFICIENT_AOI_COVERAGE = "INSUFFICIENT_AOI_COVERAGE"
    MODALITY_PAIR_INCOMPATIBLE = "MODALITY_PAIR_INCOMPATIBLE"
    DATA_SOURCE_UNAVAILABLE = "DATA_SOURCE_UNAVAILABLE"
    DATA_SOURCE_CONFIGURATION_REQUIRED = "DATA_SOURCE_CONFIGURATION_REQUIRED"
    INVALID_RASTER = "INVALID_RASTER"


class OrchestrationError(ValueError):
    def __init__(self, code: FailureCode, message: str):
        super().__init__(message); self.code = code


class Modality(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    OPTICAL_SAR = "OPTICAL_SAR"
    MULTISPECTRAL = "MULTISPECTRAL"


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _polygon_area_km2(ring: Sequence[Sequence[float]]) -> float:
    # Deterministic local equirectangular approximation; not a geodesic survey value.
    lat = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
    scale_x, scale_y = 111.32 * math.cos(math.radians(lat)), 111.32
    area = abs(sum((a[0]*scale_x)*(b[1]*scale_y) - (b[0]*scale_x)*(a[1]*scale_y) for a,b in zip(ring, ring[1:]))) / 2
    return area


@dataclass(frozen=True)
class AOI:
    crs: str
    bounds: tuple[float, float, float, float]
    geometry: dict[str, Any]
    area_km2: float | None
    input_type: str
    source: str
    named_location: str | None = None

    @classmethod
    def from_input(cls, value: dict[str, Any] | str, *, source: str = "request") -> "AOI":
        if isinstance(value, str):
            if not value.strip(): raise OrchestrationError(FailureCode.INVALID_AOI, "Named location is empty")
            raise OrchestrationError(FailureCode.INVALID_AOI, "Named locations require an explicit geocoding adapter; no coordinates were fabricated")
        if not isinstance(value, dict): raise OrchestrationError(FailureCode.INVALID_AOI, "AOI must be a point, bbox, polygon, or GeoJSON geometry")
        kind = value.get("type")
        if kind == "Feature":
            geometry=value.get("geometry"); input_type="geojson_feature"
            if not isinstance(geometry, dict): raise OrchestrationError(FailureCode.INVALID_AOI, "GeoJSON Feature requires geometry")
            kind=geometry.get("type"); value=geometry
        else: input_type=str(kind).lower()
        if kind == "point":
            lat, lon=value.get("latitude"),value.get("longitude")
            if not (_finite(lat) and _finite(lon) and -90<=lat<=90 and -180<=lon<=180): raise OrchestrationError(FailureCode.INVALID_AOI,"Point coordinates are invalid")
            geometry={"type":"Point","coordinates":[lon,lat]}; bounds=(float(lon),float(lat),float(lon),float(lat)); area=0.0
        elif kind == "Point":
            coords=value.get("coordinates")
            if not isinstance(coords,list) or len(coords)!=2: raise OrchestrationError(FailureCode.INVALID_AOI,"GeoJSON Point coordinates are invalid")
            return cls.from_input({"type":"point","longitude":coords[0],"latitude":coords[1]},source=source)
        elif kind == "bbox":
            b=value.get("bounds")
            if not isinstance(b,(list,tuple)) or len(b)!=4 or not all(_finite(x) for x in b): raise OrchestrationError(FailureCode.INVALID_AOI,"Bounding box must contain four finite coordinates")
            w,s,e,n=map(float,b)
            if not (-180<=w<e<=180 and -90<=s<n<=90): raise OrchestrationError(FailureCode.INVALID_AOI,"Bounding box is empty or outside EPSG:4326")
            ring=[[w,s],[e,s],[e,n],[w,n],[w,s]]; geometry={"type":"Polygon","coordinates":[ring]}; bounds=(w,s,e,n); area=_polygon_area_km2(ring)
        elif kind in {"Polygon","polygon"}:
            coords=value.get("coordinates")
            if not isinstance(coords,list) or len(coords)!=1 or not isinstance(coords[0],list) or len(coords[0])<4: raise OrchestrationError(FailureCode.INVALID_AOI,"Polygon requires one non-empty exterior ring")
            ring=coords[0]
            if ring[0]!=ring[-1] or any(not isinstance(p,(list,tuple)) or len(p)!=2 or not all(_finite(x) for x in p) for p in ring): raise OrchestrationError(FailureCode.INVALID_AOI,"Polygon ring must be closed with finite coordinate pairs")
            xs=[float(p[0]) for p in ring]; ys=[float(p[1]) for p in ring]
            if min(xs)<-180 or max(xs)>180 or min(ys)<-90 or max(ys)>90 or min(xs)==max(xs) or min(ys)==max(ys) or _polygon_area_km2(ring)==0: raise OrchestrationError(FailureCode.INVALID_AOI,"Polygon is empty, degenerate, or outside EPSG:4326")
            geometry={"type":"Polygon","coordinates":[[[float(x),float(y)] for x,y in ring]]}; bounds=(min(xs),min(ys),max(xs),max(ys)); area=_polygon_area_km2(ring)
        else: raise OrchestrationError(FailureCode.INVALID_AOI,"Unsupported AOI geometry type")
        return cls("EPSG:4326",bounds,geometry,round(area,6),input_type,source)


@dataclass(frozen=True)
class TemporalRequest:
    start: date
    end: date
    before: date | None = None
    after: date | None = None
    tolerance_days: int = 0

    @classmethod
    def create(cls, *, acquisition_date: str|None=None, start: str|None=None, end: str|None=None, before: str|None=None, after: str|None=None, tolerance_days: int=0):
        def parse(v):
            try: return date.fromisoformat(v) if v else None
            except (TypeError,ValueError): raise OrchestrationError(FailureCode.INVALID_TEMPORAL_REQUEST,"Dates must use valid ISO YYYY-MM-DD values") from None
        one,s,e,b,a=map(parse,(acquisition_date,start,end,before,after))
        if one: s=e=one
        if b or a:
            if not (b and a): raise OrchestrationError(FailureCode.INVALID_TEMPORAL_REQUEST,"Before and after dates must both be supplied")
            s,e=b,a
        if not (s and e) or s>e or (b and b>=a) or type(tolerance_days) is not int or tolerance_days<0:
            raise OrchestrationError(FailureCode.INVALID_TEMPORAL_REQUEST,"Temporal range is incomplete or invalid")
        return cls(s,e,b,a,tolerance_days)


@dataclass(frozen=True)
class SourceDefinition:
    key: str; satellite: str; sensor: str; modalities: tuple[Modality,...]
    bands: tuple[str,...]; native_gsd_by_band: dict[str,float]; operational_adapters: tuple[str,...]

SOURCE_REGISTRY={
    "sentinel-2":SourceDefinition("sentinel-2","Sentinel-2","MSI",(Modality.OPTICAL,Modality.MULTISPECTRAL),OPTICAL_BANDS,{**{b:20.0 for b in OPTICAL_BANDS},**{b:10.0 for b in ("B02","B03","B04","B08")},"B01":60.0,"B09":60.0},("local", "gee_if_configured")),
    "sentinel-1":SourceDefinition("sentinel-1","Sentinel-1","C-SAR",(Modality.SAR,),SAR_BANDS,{"VV":10.0,"VH":10.0},("local", "gee_if_configured")),
}


def get_source(key: str) -> SourceDefinition:
    try:return SOURCE_REGISTRY[key.lower()]
    except (KeyError,AttributeError):raise OrchestrationError(FailureCode.UNSUPPORTED_SOURCE,f"Unsupported or unverified source: {key}") from None


@dataclass(frozen=True)
class ResolutionContract:
    requested_gsd: float; native_gsd: float; source_product_gsd: float
    processing_gsd: float; resampled_gsd: float|None; resolution_status: str; information_status: str

    @classmethod
    def evaluate(cls, requested: float, native: float, *, processing: float|None=None):
        if not all(_finite(v) and v>0 for v in (requested,native)) or (processing is not None and (not _finite(processing) or processing<=0)):
            raise ValueError("GSD values must be finite and positive")
        processing=float(native if processing is None else processing)
        status="native_compatible" if requested>=native else "unavailable_at_requested_resolution"
        info="native_information" if processing==native else f"resampled_from_{native:g}m"
        if processing< native and info=="native_information": raise ValueError("Upsampling cannot be claimed as native information")
        return cls(float(requested),float(native),float(native),processing,processing if processing!=native else None,status,info)


@dataclass(frozen=True)
class SceneCandidate:
    scene_id:str; source:str; satellite:str; sensor:str; modality:Modality; acquisition_time:datetime
    native_gsd:float; crs:str; bounds:tuple[float,float,float,float]; coverage_fraction:float
    band_order:tuple[str,...]; cloud_percent:float|None=None; product_type:str|None=None
    access:dict[str,Any]=field(default_factory=dict); suitability:tuple[str,...]=()

    def validate(self):
        definition=get_source(self.source)
        if self.modality not in definition.modalities or not self.scene_id or self.acquisition_time.tzinfo is None: raise ValueError("Invalid scene identity, modality, or timezone")
        if not 0<=self.coverage_fraction<=1 or not self.crs or len(self.bounds)!=4 or not all(_finite(x) for x in self.bounds): raise ValueError("Invalid scene coverage or spatial metadata")
        if self.cloud_percent is not None and not 0<=self.cloud_percent<=100: raise ValueError("Invalid cloud percentage")
        if tuple(self.band_order)!=definition.bands: raise ValueError("Scene band order does not match the registered source contract")


@dataclass(frozen=True)
class AvailabilityRequest:
    aoi:AOI; temporal:TemporalRequest; modality:Modality; requested_gsd:float
    source_preferences:tuple[str,...]=(); max_cloud_percent:float|None=None

@dataclass(frozen=True)
class AvailabilityResponse:
    status:str; candidates:tuple[SceneCandidate,...]; provider:str; error_code:FailureCode|None=None; message:str=""


class DataSourceAdapter(Protocol):
    name:str
    def search(self, request:AvailabilityRequest)->AvailabilityResponse: ...


def filter_candidates(request:AvailabilityRequest, candidates:Sequence[SceneCandidate])->tuple[SceneCandidate,...]:
    result=[]
    for scene in candidates:
        scene.validate()
        if request.source_preferences and scene.source not in request.source_preferences: continue
        if scene.modality not in ({Modality.OPTICAL,Modality.MULTISPECTRAL} if request.modality in {Modality.OPTICAL,Modality.MULTISPECTRAL} else {Modality.SAR}): continue
        d=scene.acquisition_time.date()
        if not request.temporal.start<=d<=request.temporal.end or scene.coverage_fraction<1: continue
        if request.max_cloud_percent is not None and scene.cloud_percent is not None and scene.cloud_percent>request.max_cloud_percent: continue
        result.append(scene)
    return tuple(result)


def select_scene(request:AvailabilityRequest, candidates:Sequence[SceneCandidate])->tuple[SceneCandidate,str]:
    eligible=filter_candidates(request,candidates)
    if not eligible: raise OrchestrationError(FailureCode.NO_SCENE_AVAILABLE,"No candidate satisfies AOI, date, modality, coverage, and cloud constraints")
    middle=request.temporal.start+(request.temporal.end-request.temporal.start)/2
    ranked=sorted(eligible,key=lambda s:(s.native_gsd>request.requested_gsd,abs((s.acquisition_time.date()-middle).days),s.cloud_percent if s.cloud_percent is not None else 101,s.scene_id))
    scene=ranked[0]; reason="Full AOI coverage; deterministic nearest-date ordering; cloud constraint satisfied; resolution suitability recorded without native-detail inflation."
    return scene,reason


class LocalSceneAdapter:
    name="local"
    def __init__(self, scenes:Sequence[SceneCandidate]): self.scenes=tuple(scenes)
    def search(self,request):
        candidates=filter_candidates(request,self.scenes)
        return AvailabilityResponse("available" if candidates else "unavailable",candidates,self.name,None if candidates else FailureCode.NO_SCENE_AVAILABLE)


@dataclass(frozen=True)
class StagedRaster:
    path:Path; scene:SceneCandidate; modality:Modality; file_sha256:str

class AcquisitionAdapter(Protocol):
    def acquire(self,scene:SceneCandidate)->tuple[StagedRaster,...]: ...

class LocalAcquisitionAdapter:
    def acquire(self,scene):
        paths=scene.access.get("paths")
        if not isinstance(paths,dict): raise OrchestrationError(FailureCode.DATA_SOURCE_CONFIGURATION_REQUIRED,"Local scene paths are not configured")
        staged=[]
        for modality,path in paths.items():
            p=Path(path)
            if not p.is_file(): raise OrchestrationError(FailureCode.INVALID_RASTER,f"Staged raster does not exist: {p}")
            h=hashlib.sha256(p.read_bytes()).hexdigest(); staged.append(StagedRaster(p,scene,Modality(modality),h))
        return tuple(staged)


@dataclass(frozen=True)
class ValidatedSatelliteDataset:
    request_id:str; scene_ids:tuple[str,...]; optical:np.ndarray|None; sar:np.ndarray|None
    aoi:AOI; crs:str; bounds:tuple[float,float,float,float]; native_gsd:float; processing_gsd:float
    band_orders:dict[str,tuple[str,...]]; acquisition_times:tuple[str,...]; source_hashes:dict[str,str]; provenance:dict[str,Any]


def _covers(scene_bounds,aoi_bounds):
    w,s,e,n=scene_bounds; aw,as_,ae,an=aoi_bounds
    return w<=aw and s<=as_ and e>=ae and n>=an


def validate_raster(path:Path,scene:SceneCandidate,aoi:AOI,expected_modality:Modality)->dict[str,Any]:
    try:
        with rasterio.open(path) as ds:
            if not ds.crs or ds.width<1 or ds.height<1 or ds.transform.a<=0 or ds.transform.e>=0: raise ValueError("CRS, dimensions, or north-up transform is invalid")
            expected=OPTICAL_BANDS if expected_modality in {Modality.OPTICAL,Modality.MULTISPECTRAL} else SAR_BANDS
            descriptions=tuple(x for x in ds.descriptions if x)
            if ds.count!=len(expected) or descriptions!=expected: raise ValueError(f"Expected exact band order {expected}; B10 is excluded")
            values=ds.read(out_dtype="float32",masked=True).filled(np.nan)
            if not np.isfinite(values).all() or not (ds.read_masks()==255).all(): raise ValueError("Raster contains masked or non-finite pixels")
            geographic=transform_bounds(ds.crs,"EPSG:4326",*ds.bounds,densify_pts=21)
            if not _covers(geographic,aoi.bounds): raise OrchestrationError(FailureCode.INSUFFICIENT_AOI_COVERAGE,"Raster does not fully cover the requested AOI")
            if scene.crs!=str(ds.crs) or not np.allclose(scene.bounds,ds.bounds,atol=1e-6,rtol=0): raise ValueError("Scene metadata does not match staged raster")
            return {"array":values,"crs":str(ds.crs),"bounds":tuple(ds.bounds),"resolution":tuple(map(abs,ds.res)),"transform":tuple(ds.transform),"band_order":expected}
    except OrchestrationError: raise
    except (rasterio.errors.RasterioError,ValueError,OSError) as exc: raise OrchestrationError(FailureCode.INVALID_RASTER,str(exc)) from None


def validate_spatial_pair(optical:dict[str,Any],sar:dict[str,Any]):
    for key,tolerance in (("bounds",1e-6),("resolution",1e-8),("transform",1e-8)):
        if not np.allclose(optical[key],sar[key],atol=tolerance,rtol=0): raise OrchestrationError(FailureCode.MODALITY_PAIR_INCOMPATIBLE,f"Optical/SAR {key} mismatch")
    if optical["crs"]!=sar["crs"] or optical["array"].shape[1:]!=sar["array"].shape[1:]: raise OrchestrationError(FailureCode.MODALITY_PAIR_INCOMPATIBLE,"Optical/SAR CRS or grid-shape mismatch")


@dataclass(frozen=True)
class SatQueryRequest:
    query:str; aoi:AOI; temporal:TemporalRequest; modality:Modality; requested_gsd:float
    satellite_preferences:tuple[str,...]=(); max_cloud_percent:float|None=None; before_after_required:bool=False; request_id:str=field(default_factory=lambda:str(uuid4()))

@dataclass(frozen=True)
class ExecutionPlan:
    request:SatQueryRequest; sources:tuple[str,...]; validation_steps:tuple[str,...]; created_at:str

@dataclass(frozen=True)
class ProcessingResult:
    request_id:str; status:str; selected_scenes:tuple[str,...]; phase2_input:Phase2PipelineInput|None
    advanced_features:np.ndarray|None; provenance:dict[str,Any]; warnings:tuple[str,...]=(); errors:tuple[dict[str,str],...]=()


def bridge_to_phase2(dataset:ValidatedSatelliteDataset,croma_model,physical_provider,*,croma_provenance:dict[str,Any],split="test")->Phase2PipelineInput:
    if dataset.optical is None or dataset.sar is None: raise OrchestrationError(FailureCode.MODALITY_PAIR_INCOMPATIBLE,"Phase 1 paired foundation requires optical and SAR tensors")
    identity=SampleIdentity(dataset.request_id,dataset.scene_ids[1],dataset.scene_ids[0],split,{"acquisition_times":list(dataset.acquisition_times)},dataset.crs,dataset.bounds,(dataset.processing_gsd,dataset.processing_gsd),dataset.provenance)
    batch=prepare_batch(torch.from_numpy(dataset.optical[None]),torch.from_numpy(dataset.sar[None]),[identity])
    with torch.inference_mode(): features=croma_model(optical_images=batch.optical,SAR_images=batch.sar)
    vector,report=physical_provider.extract(dataset.optical,dataset.sar,list(OPTICAL_BANDS),list(SAR_BANDS)); report={**report,"sample_id":identity.patch_id}
    return adapt_phase1_output(batch,features,np.asarray(vector,dtype=np.float32)[None],(report,),croma_provenance=croma_provenance,experiment_configuration={"phase3_version":PHASE3_VERSION},source_hashes=(dataset.source_hashes,))

