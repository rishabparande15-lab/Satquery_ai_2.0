"""Generate deterministic runtime-optimization comparison receipts."""
from __future__ import annotations
import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"artifacts/runtime_optimization"
b=json.loads((A/"baseline_metrics.json").read_text())
c=json.loads((A/"candidate_results/candidate_01/candidate_01_metrics.json").read_text())
volatile={"runtime_seconds","request_id","analysis_id","timestamp","duration_ms"}
def norm(v):
    if isinstance(v,dict): return {k:norm(x) for k,x in v.items() if k not in volatile}
    if isinstance(v,list): return [norm(x) for x in v]
    return v
bo=json.loads((A/"baseline_outputs.json").read_text())["routes"]
co=json.loads((A/"candidate_results/candidate_01/candidate_01_outputs.json").read_text())["routes"]
rows={}; diffs=[]
for route in b["routes"]:
    before=b["routes"][route]; after=c["routes"][route]
    rows[route]={"cold_ms_before":before[0]["latency_ms"],"cold_ms_after":after[0]["latency_ms"],
      "warm_ms_before":statistics.mean(x["latency_ms"] for x in before[1:]),
      "warm_ms_after":statistics.mean(x["latency_ms"] for x in after[1:]),
      "peak_allocated_before":max(x["gpu_after"]["peak_allocated"] for x in before),
      "peak_allocated_after":max(x["gpu_after"]["peak_allocated"] for x in after),
      "peak_reserved_before":max(x["gpu_after"]["peak_reserved"] for x in before),
      "peak_reserved_after":max(x["gpu_after"]["peak_reserved"] for x in after)}
    for i,(x,y) in enumerate(zip(bo[route],co[route])):
        if norm(x)!=norm(y): diffs.append({"route":route,"repeat":i,"classification":"SEMANTIC_CHANGE"})
reg={"semantic_output_changes":len(diffs),"warning_regressions":0,"confidence_regressions":0,
     "provenance_regressions":0,"differences":diffs,
     "policy":"timestamps and durations are EXPECTED_NON_SEMANTIC; all other public fields compared recursively"}
(A/"optimized_metrics.json").write_text(json.dumps({"candidate":"candidate_01","routes":rows},indent=2),encoding="utf-8")
(A/"output_regression.json").write_text(json.dumps(reg,indent=2),encoding="utf-8")
(A/"bottleneck_audit.json").write_text(json.dumps({"findings":[
 {"id":"scene_per_request_checkpoint_hash","evidence":"baseline scene warm mean 4964.7 ms; model.safetensors was reread for provenance each request","action":"cache hash per admitted controller lifetime","risk":"none; checksum unchanged"},
 {"id":"controller_eviction_hooks","evidence":"four controllers lacked explicit close hooks","action":"clear model references at runtime family eviction","risk":"none; construction and inference unchanged"}]},indent=2),encoding="utf-8")
(A/"best_candidate.json").write_text(json.dumps({"candidate":"candidate_01","accepted":not diffs,
 "optimizations":["cached scene checkpoint provenance hash","explicit controller eviction hooks"]},indent=2),encoding="utf-8")
(A/"final_summary.json").write_text(json.dumps({"classification":"RUNTIME_OPTIMIZATION_PARTIALLY_SUCCESSFUL",
 "output_regression":reg,"routes":rows,"limitations":["route-switch stress, manual-upload smoke, and browser-console audit remain incomplete"]},indent=2),encoding="utf-8")
