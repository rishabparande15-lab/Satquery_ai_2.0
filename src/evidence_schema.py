"""Deterministic, research-only spatial evidence representation.

The module creates traceable evidence objects; it does not perform semantic
segmentation, confidence calibration, or natural-language generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pixel_features import aggregate_tokens, candidate_maps, edge_strength, local_mean_std

SCHEMA_VERSION = "spatial_evidence_v1"
MAPPING_VERSION = "north_up_row_major_120_to_15_v1"
FEATURE_VERSION = "pixel_features_v1"
STRENGTHS = ("STRONG", "MODERATE", "WEAK", "UNAVAILABLE")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def token_extent(index: int, transform, *, image_size: int = 120, grid_size: int = 15) -> dict:
    if not 0 <= index < grid_size * grid_size or image_size % grid_size:
        raise ValueError("invalid token index or grid")
    t = list(transform)
    if len(t) != 9 or t[1] != 0 or t[3] != 0 or t[0] <= 0 or t[4] >= 0:
        raise ValueError("only finite north-up affine transforms are supported")
    block = image_size // grid_size; row, col = divmod(index, grid_size)
    r0, r1, c0, c1 = row*block, (row+1)*block, col*block, (col+1)*block
    left, right = t[2]+c0*t[0], t[2]+c1*t[0]
    top, bottom = t[5]+r0*t[4], t[5]+r1*t[4]
    return {"token_index":index,"grid_position":[row,col],"pixel_bounds":[c0,r0,c1,r1],"block_size":[block,block],
            "map_bounds":[left,bottom,right,top]}


def group_regions(token_indices, *, grid_size: int = 15) -> list[list[int]]:
    """Four-connected deterministic evidence regions, sorted by first token."""
    remaining=set(int(i) for i in token_indices)
    if any(i < 0 or i >= grid_size*grid_size for i in remaining): raise ValueError("token out of range")
    groups=[]
    while remaining:
        seed=min(remaining); remaining.remove(seed); stack=[seed]; group=[]
        while stack:
            item=stack.pop(); group.append(item); row,col=divmod(item,grid_size)
            neighbors=[]
            if row: neighbors.append(item-grid_size)
            if row+1<grid_size: neighbors.append(item+grid_size)
            if col: neighbors.append(item-1)
            if col+1<grid_size: neighbors.append(item+1)
            for neighbor in sorted(neighbors, reverse=True):
                if neighbor in remaining: remaining.remove(neighbor); stack.append(neighbor)
        groups.append(sorted(group))
    return sorted(groups,key=lambda x:x[0])


def evidence_strength(*, coverage: float, support_fraction: float, largest_region_fraction: float) -> str:
    """Transparent descriptive strength; explicitly not statistical confidence."""
    if coverage < .5: return "UNAVAILABLE"
    if support_fraction >= .35 and largest_region_fraction >= .2: return "STRONG"
    if support_fraction >= .15 and largest_region_fraction >= .08: return "MODERATE"
    if support_fraction > 0: return "WEAK"
    return "UNAVAILABLE"


def _feature_record(name, values, unit, sensor, source_artifact, aggregation="mean"):
    tokens, stats=aggregate_tokens(values); column=stats.index(aggregation)
    valid_count=np.isfinite(values).reshape(15,8,15,8).transpose(0,2,1,3).reshape(225,64).sum(1)
    return tokens[:,column], [{"feature_id":name,"value":float(tokens[i,column]) if np.isfinite(tokens[i,column]) else None,
        "unit":unit,"sensor":sensor,"evidence_type":"DERIVED","aggregation":aggregation,
        "valid_pixel_count":int(valid_count[i]),"valid_ratio":float(tokens[i,-1]),"feature_version":FEATURE_VERSION,
        "source_artifact":source_artifact} for i in range(225)]


def _claim(claim_id, text, feature_id, sensor, values, tokens, regions, source_artifact, coverage):
    if not tokens: return None
    largest=max(map(len,regions),default=0)/225
    strength=evidence_strength(coverage=coverage,support_fraction=len(tokens)/225,largest_region_fraction=largest)
    support={"feature_id":feature_id,"sensor":sensor,"evidence_type":"DERIVED","token_indices":tokens,
             "values":[float(values[i]) for i in tokens],"source_artifact":source_artifact}
    return {"claim_id":claim_id,"claim":text,"claim_type":"INFERRED","sensor_view":sensor,"strength":strength,
            "confidence_status":"uncalibrated; evidence strength is descriptive","support":[support],
            "region_ids":[f"{claim_id}-region-{j:03d}" for j in range(len(regions))]}


def build_spatial_evidence(*, sample_id: str, optical: np.ndarray | None, sar: np.ndarray | None,
        transform, crs: str, scene_metadata: Mapping[str,Any], provenance: Mapping[str,Any],
        croma: Mapping[str,Mapping[str,Any]] | None=None, predictions: Mapping[int,list[dict]] | None=None) -> dict:
    if optical is None and sar is None: raise ValueError("at least one sensor is required")
    maps={}; specs={}; optical_order=("B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12")
    sar_order=("VV","VH")
    if optical is not None:
        if np.asarray(optical).shape != (12,120,120): raise ValueError("optical must be [12,120,120]")
        o={b:np.asarray(optical[i],dtype=np.float32) for i,b in enumerate(optical_order)}
        # Candidate helper needs SAR; use only the optical subset formulas here.
        from .pixel_features import safe_normalized_difference
        maps["NDVI"]=safe_normalized_difference(o["B08"],o["B04"])[0]; maps["NDWI"]=safe_normalized_difference(o["B03"],o["B08"])[0]
        maps["MNDWI"]=safe_normalized_difference(o["B03"],o["B11"])[0]; maps["NDBI"]=safe_normalized_difference(o["B11"],o["B08"])[0]
        maps["BSI"]=safe_normalized_difference(o["B11"]+o["B04"],o["B08"]+o["B02"])[0]
        maps["NDVI_local_std_5"]=local_mean_std(maps["NDVI"],5)[1]; maps["NDVI_edge_strength"]=edge_strength(maps["NDVI"])
        for n in ("NDVI","NDWI","MNDWI","NDBI","BSI"): specs[n]=(n,"unitless","OPTICAL")
        specs.update(NDVI_local_std_5=("NDVI local std 5x5","unitless","OPTICAL"),NDVI_edge_strength=("NDVI edge strength","unitless/pixel","OPTICAL"))
    if sar is not None:
        if np.asarray(sar).shape != (2,120,120): raise ValueError("sar must be [2,120,120]")
        maps["VV"]=np.asarray(sar[0],dtype=np.float32); maps["VH"]=np.asarray(sar[1],dtype=np.float32); maps["VV_minus_VH"]=maps["VV"]-maps["VH"]
        maps["VV_local_std_5"]=local_mean_std(maps["VV"],5)[1]
        specs.update(VV=("VV","dB","SAR"),VH=("VH","dB","SAR"),VV_minus_VH=("VV minus VH","dB","SAR"),VV_local_std_5=("VV local std 5x5","dB","SAR"))
    source_artifact=str(provenance.get("pixel_source_artifact","UNAVAILABLE")); feature_values={}; feature_records={}
    for key,(_,unit,sensor) in specs.items(): feature_values[key],feature_records[key]=_feature_record(key,maps[key],unit,sensor,source_artifact)
    croma=croma or {}; predictions=predictions or {}; tokens=[]
    for i in range(225):
        latent={}
        for mode in ("optical","sar","joint"):
            ref=croma.get(mode)
            latent[mode]=({"present":True,"semantic_status":"LATENT; dimensions have no assigned meaning","artifact":dict(ref),"token_row":i}
                          if ref else {"present":False,"status":"UNAVAILABLE"})
        tokens.append({**token_extent(i,transform),"valid_pixel_ratio":float(min((feature_records[k][i]["valid_ratio"] for k in feature_records),default=0)),
            "features":{k:feature_records[k][i] for k in feature_records},"croma":latent,
            "predictions":predictions.get(i,[]),"prediction_status":"MODEL_PREDICTED" if predictions.get(i) else "UNAVAILABLE"})
    region_objects=[]; claims=[]
    rules=[]
    if "NDVI" in feature_values: rules += [("vegetation","Vegetation-related spectral evidence is spatially concentrated.","NDVI","OPTICAL",.2)]
    if "NDWI" in feature_values: rules += [("water","Water-related spectral evidence is spatially concentrated.","NDWI","OPTICAL",0.)]
    if "NDBI" in feature_values: rules += [("built_up","Built-up-related spectral evidence occurs in localized tokens.","NDBI","OPTICAL",0.)]
    if "VV_minus_VH" in feature_values: rules += [("sar_polarization","Elevated SAR polarization-contrast evidence is spatially concentrated.","VV_minus_VH","SAR",-np.inf)]
    for cid,text,key,sensor,floor in rules:
        values=feature_values[key]; finite=values[np.isfinite(values)]; threshold=max(float(np.percentile(finite,75)),floor) if finite.size else np.inf
        selected=[int(i) for i,v in enumerate(values) if np.isfinite(v) and v>=threshold and v>floor]
        groups=group_regions(selected)
        for j,group in enumerate(groups):
            region_objects.append({"region_id":f"{cid}-region-{j:03d}","kind":"DETERMINISTIC_EVIDENCE_REGION","feature_id":key,
                "sensor":sensor,"token_indices":group,"token_count":len(group),"threshold_rule":f">= sample token p75 ({threshold:.8g}) and > {floor}",
                "spatial_extents":[token_extent(i,transform) for i in group],"limitations":"Four-connected threshold grouping; not learned segmentation."})
        claim=_claim(cid,text,key,sensor,values,selected,groups,source_artifact,float(np.mean([t["valid_pixel_ratio"] for t in tokens])))
        if claim: claims.append(claim)
    prediction_scores={}
    for token in tokens:
        for prediction in token["predictions"]:
            if "class" not in prediction or "score" not in prediction or not np.isfinite(prediction["score"]):
                raise ValueError("prediction requires finite class/score")
            prediction_scores.setdefault(str(prediction["class"]),[]).append(float(prediction["score"]))
    prediction_summary={"status":"UNAVAILABLE","reason":"No trained task-head prediction exists for sample 61_39"}
    if prediction_scores:
        aggregate=sorted(({"class":name,"mean_token_score":float(np.mean(scores)),"token_count":len(scores)} for name,scores in prediction_scores.items()),key=lambda x:(-x["mean_token_score"],x["class"]))
        prediction_summary={"status":"MODEL_PREDICTED","aggregation":"mean available token scores by class","dominant_class":aggregate[0]["class"],"classes":aggregate}
    scene_summary={"token_count":225,"valid_data_coverage":float(np.mean([t["valid_pixel_ratio"] for t in tokens])),
        "feature_summaries":{k:{"median":float(np.nanmedian(v)),"p25":float(np.nanpercentile(v,25)),"p75":float(np.nanpercentile(v,75))} for k,v in feature_values.items()},
        "prediction_summary":prediction_summary,
        "croma_status":{m:("LATENT_PRESENT" if croma.get(m) else "UNAVAILABLE") for m in ("optical","sar","joint")}}
    result={"schema_version":SCHEMA_VERSION,"scene":{"sample_id":sample_id,"crs":crs,"transform":list(transform),"image_shape":[120,120],
        "token_grid":[15,15],"scene_metadata":dict(scene_metadata),"summary":scene_summary},"tokens":tokens,"regions":region_objects,"claims":claims,
        "sensor_views":{"optical":{"status":"AVAILABLE" if optical is not None else "UNAVAILABLE","feature_ids":[k for k,v in specs.items() if v[2]=="OPTICAL"]},
            "sar":{"status":"AVAILABLE" if sar is not None else "UNAVAILABLE","feature_ids":[k for k,v in specs.items() if v[2]=="SAR"]},
            "joint":{"status":"LATENT_PRESENT_NO_SEMANTIC_CLAIM" if croma.get("joint") else "UNAVAILABLE","basis":"joint CROMA only; concatenation does not create a joint claim"}},
        "provenance":{**dict(provenance),"schema_version":SCHEMA_VERSION,"feature_version":FEATURE_VERSION,"spatial_mapping_version":MAPPING_VERSION}}
    validate_evidence(result); return result


def validate_evidence(value: Mapping[str,Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION or len(value.get("tokens",[])) != 225: raise ValueError("invalid evidence schema/token count")
    tokens=value["tokens"]
    if [t["token_index"] for t in tokens] != list(range(225)): raise ValueError("tokens are not row-major")
    region_ids={r["region_id"] for r in value.get("regions",[])}
    for claim in value.get("claims",[]):
        if claim.get("strength") not in STRENGTHS or not claim.get("support"): raise ValueError("unsupported claim")
        if not set(claim.get("region_ids",[])) <= region_ids: raise ValueError("claim references missing region")
        for support in claim["support"]:
            if not support.get("source_artifact") or support["source_artifact"]=="UNAVAILABLE": raise ValueError("claim is not traceable")
            if len(support.get("token_indices",[])) != len(support.get("values",[])): raise ValueError("support mismatch")


@dataclass(frozen=True)
class EvidenceQueries:
    evidence: Mapping[str,Any]
    def get_scene_summary(self): return self.evidence["scene"]["summary"]
    def get_token_evidence(self,index):
        if not 0 <= index < 225: raise IndexError(index)
        return self.evidence["tokens"][index]
    def get_feature_map_evidence(self,feature): return [t["features"][feature] for t in self.evidence["tokens"] if feature in t["features"]] or {"status":"UNAVAILABLE"}
    def get_class_regions(self,class_name):
        stored=[r for r in self.evidence["regions"] if r["feature_id"]==class_name or r.get("class_name")==class_name]
        if stored: return stored
        indices=[t["token_index"] for t in self.evidence["tokens"] if any(p.get("class")==class_name for p in t["predictions"])]
        return [{"kind":"DETERMINISTIC_EVIDENCE_REGION","class_name":class_name,"token_indices":g,
                 "limitations":"Connectivity grouping of supplied model predictions; not segmentation."} for g in group_regions(indices)]
    def get_sensor_evidence(self,sensor): return self.evidence["sensor_views"].get(sensor.lower(),{"status":"UNAVAILABLE"})
    def get_supported_claims(self): return list(self.evidence["claims"])
