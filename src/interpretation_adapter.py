"""Allowlisted evidence-to-text adapter; no free-form generation or inference."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from .evidence_schema import validate_evidence

ADAPTER_VERSION = "constrained_interpretation_v1"
UNAVAILABLE = "The current SatQuery pipeline does not provide validated evidence for that question."
ALLOWED = {
    "vegetation": {"sensor":"OPTICAL","feature":"NDVI","subject":"vegetation-related spectral evidence",
                   "claim":"Vegetation-related spectral evidence is spatially concentrated."},
    "water": {"sensor":"OPTICAL","feature":"NDWI","subject":"water-related spectral evidence",
              "claim":"Water-related spectral evidence is spatially concentrated."},
    "sar_polarization": {"sensor":"SAR","feature":"VV_minus_VH","subject":"polarization-related surface variation",
                         "claim":"Elevated SAR polarization-contrast evidence is spatially concentrated."},
    "heterogeneity": {"sensor":"OPTICAL","feature":"NDVI_local_std_5","subject":"spatial heterogeneity in the available evidence",
                      "claim":"Spatial heterogeneity is present in the available evidence."},
}
FORBIDDEN_QUESTION_TERMS = re.compile(r"\b(houses?|cars?|vehicles?|roads?|company|companies|owner|exact\s+(?:count|identity|address)|how many)\b",re.I)


def _route(question: str) -> dict:
    q=" ".join(str(question).lower().split())
    if not q or FORBIDDEN_QUESTION_TERMS.search(q): return {"status":"UNSUPPORTED","claim_ids":[]}
    spatial=bool(re.search(r"\b(where|location|located|region)\b",q))
    if "vegetation" in q: ids=["vegetation"]
    elif "water" in q or "river" in q: ids=["water"]
    elif "sar" in q or "radar" in q: ids=["sar_polarization"]
    elif "homogeneous" in q or "heterogeneous" in q or "heterogeneity" in q: ids=["heterogeneity"]
    elif "support" in q or "evidence" in q or "present" in q or "show" in q or "describe" in q or "analy" in q: ids=list(ALLOWED)
    else: return {"status":"UNSUPPORTED","claim_ids":[]}
    return {"status":"SUPPORTED","claim_ids":ids,"spatial_requested":spatial,"technical_requested":bool(re.search(r"\b(technical|details?|tokens?|provenance)\b",q))}


def _region_position(region: Mapping[str,Any]) -> str | None:
    extents=region.get("spatial_extents") or []
    positions=[x.get("grid_position") for x in extents if isinstance(x,dict) and len(x.get("grid_position",[]))==2]
    if not positions: return None
    row=sum(x[0] for x in positions)/len(positions); col=sum(x[1] for x in positions)/len(positions)
    vertical="upper" if row<5 else "lower" if row>=10 else "central"
    horizontal="left" if col<5 else "right" if col>=10 else "central"
    if vertical==horizontal=="central": return "central portion"
    if horizontal=="central": return f"{vertical} portion"
    if vertical=="central": return f"{horizontal} portion"
    return f"{vertical}-{horizontal} portion"


def _accepted_claims(evidence: Mapping[str,Any]) -> tuple[list[dict],list[dict]]:
    regions={r["region_id"]:r for r in evidence.get("regions",[])}; accepted=[]; rejected=[]
    for claim in evidence.get("claims",[]):
        cid=claim.get("claim_id"); rule=ALLOWED.get(cid); reasons=[]
        if not rule: reasons.append("claim type is not allowlisted")
        if rule and claim.get("claim")!=rule["claim"]: reasons.append("claim wording does not match the validated contract")
        if rule and claim.get("claim_type")!="INFERRED": reasons.append("epistemic type is not supported")
        if rule and rule["sensor"] and claim.get("sensor_view")!=rule["sensor"]: reasons.append("sensor does not match claim contract")
        confidence=str(claim.get("confidence_status","")).lower()
        if not confidence.startswith("uncalibrated"): reasons.append("confidence wording is not the uncalibrated contract")
        support=claim.get("support") or []
        if rule and rule["feature"] and not any(s.get("feature_id")==rule["feature"] for s in support): reasons.append("required feature is absent")
        if any(rid not in regions for rid in claim.get("region_ids",[])): reasons.append("spatial reference is missing")
        if reasons: rejected.append({"claim_id":cid,"reasons":reasons}); continue
        accepted.append(claim)
    return accepted,rejected


def _render_claim(claim: Mapping[str,Any], regions: Mapping[str,Mapping[str,Any]], *, technical: bool, spatial_requested: bool) -> tuple[str,dict]:
    cid=claim["claim_id"]; rule=ALLOWED[cid]; strength=claim["strength"].lower(); subject=rule["subject"]
    base=(f"The derived optical evidence shows {strength} {subject}." if claim["sensor_view"]=="OPTICAL" else
          f"The derived SAR evidence shows {strength} {subject}.")
    region_ids=claim.get("region_ids",[]); positions=[]
    for rid in region_ids:
        pos=_region_position(regions[rid])
        if pos and pos not in positions: positions.append(pos)
    spatial=""
    if spatial_requested and region_ids:
        primary=max((regions[rid] for rid in region_ids),key=lambda r:(r.get("token_count",0),-region_ids.index(r["region_id"])))
        position=_region_position(primary)
        spatial=f" It is concentrated in {len(region_ids)} deterministic evidence region{'s' if len(region_ids)!=1 else ''}"
        if position: spatial+=f", with the largest in the {position}"
        spatial+="."
    text=base+spatial
    token_indices=sorted({int(i) for s in claim["support"] for i in s.get("token_indices",[])})
    features=sorted({s["feature_id"] for s in claim["support"]}); artifacts=sorted({s["source_artifact"] for s in claim["support"]})
    if technical:
        text+=f" Supporting feature: {', '.join(features)}; tokens: {token_indices}; region IDs: {region_ids}. Evidence strength is descriptive, not calibrated confidence."
    trace={"claim_id":cid,"text":text,"epistemic_status":"INFERRED_FROM_DERIVED_EVIDENCE","strength":claim["strength"],
           "confidence_status":claim["confidence_status"],"sensor":claim["sensor_view"],"source_features":features,
           "token_indices":token_indices,"region_ids":region_ids,"region_positions":positions,
           "region_extents":{rid:regions[rid].get("spatial_extents",[]) for rid in region_ids},"source_artifacts":artifacts}
    return text,trace


def interpret_evidence(evidence: Mapping[str,Any] | None, question: str, *, mode: str="simple") -> dict:
    if mode not in {"simple","technical"}: raise ValueError("mode must be simple or technical")
    route=_route(question)
    base={"adapter_version":ADAPTER_VERSION,"mode":mode,"question":question,"routing":route,"calibration_status":"uncalibrated","generation":"deterministic templates; no LLM/VLM"}
    if evidence is None or route["status"]=="UNSUPPORTED":
        result={**base,"status":"UNAVAILABLE","answer":UNAVAILABLE,"claims":[],"rejected_claims":[],"provenance":{},"traceability":"no scientific claim emitted"}
        result["semantic_sha256"]=_semantic_hash(result)
        return result
    validate_evidence(evidence); accepted,rejected=_accepted_claims(evidence); wanted=set(route["claim_ids"])
    selected=[c for c in accepted if c["claim_id"] in wanted]; regions={r["region_id"]:r for r in evidence["regions"]}
    technical=mode=="technical" or route.get("technical_requested",False); rendered=[_render_claim(c,regions,technical=technical,spatial_requested=route.get("spatial_requested",False)) for c in selected]
    if not rendered:
        answer=UNAVAILABLE; status="UNAVAILABLE"
    else:
        answer=" ".join(x[0] for x in rendered); status="ANSWERED"
        sensors={x[1]["sensor"] for x in rendered}
        if len(sensors)>1: answer+=" Optical and SAR evidence provide complementary information; this is not a joint semantic prediction."
    provenance={"sample_id":evidence["scene"]["sample_id"],"evidence_schema_version":evidence["schema_version"],
        "adapter_version":ADAPTER_VERSION,"evidence_provenance":evidence.get("provenance",{})}
    result={**base,"status":status,"answer":answer,"claims":[x[1] for x in rendered],"rejected_claims":rejected,"provenance":provenance,
            "traceability":"PASS" if rendered else "no scientific claim emitted"}
    result["semantic_sha256"]=_semantic_hash(result)
    return result


def technical_companion(evidence: Mapping[str,Any] | None, question: str) -> dict:
    return interpret_evidence(evidence,question,mode="technical")


def _semantic_hash(result: Mapping[str, Any]) -> str:
    fields=("adapter_version","mode","question","routing","calibration_status","generation","status",
            "answer","claims","rejected_claims","provenance","traceability")
    semantic={k:result[k] for k in fields if k in result}
    payload=json.dumps(semantic,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_interpretation(result: Mapping[str, Any]) -> None:
    """Fail closed if an emitted sentence cannot be traced to structured evidence."""
    if result.get("status") not in {"ANSWERED","UNAVAILABLE"}: raise ValueError("invalid interpretation status")
    claims=result.get("claims")
    if not isinstance(claims,list): raise ValueError("claims must be a list")
    if result.get("status")=="UNAVAILABLE":
        if claims or result.get("answer")!=UNAVAILABLE: raise ValueError("unavailable response emitted a claim")
        return
    answer=str(result.get("answer", ""))
    if not claims: raise ValueError("answered response has no traceable claims")
    for claim in claims:
        if claim.get("claim_id") not in ALLOWED: raise ValueError("non-allowlisted claim")
        if claim.get("text") not in answer: raise ValueError("claim text is absent from answer")
        if not claim.get("source_features") or not claim.get("source_artifacts"): raise ValueError("claim lacks feature provenance")
        if claim.get("epistemic_status")!="INFERRED_FROM_DERIVED_EVIDENCE": raise ValueError("invalid epistemic status")
    expected=_semantic_hash(result)
    if result.get("semantic_sha256")!=expected: raise ValueError("semantic hash mismatch")
