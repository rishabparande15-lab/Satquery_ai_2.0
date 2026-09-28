"""User-facing, deterministic summaries over controller results."""
from typing import Any
from .controller import AnalysisResult

def summarize(result: AnalysisResult) -> dict[str, Any]:
    blocked = result.capability_status.value == "BLOCKED"
    return {"execution": {"execution_id":result.execution_id,"task":result.plan.understanding.task_type.value,"status":result.capability_status.value,"tools":([result.plan.selected_tool] if result.plan.selected_tool else []),"evidence_count":len(result.evidence),"fallback":result.audit_summary["fallback"]},
      "audit": {"validation":"PASS" if not any(x.status.value=="FAILED" for x in result.trace.steps) else "FAILED","capability":"BLOCKED" if blocked else result.capability_status.value,"evidence":len(result.evidence),"provenance":"COMPLETE" if result.provenance else "INCOMPLETE","fallback":result.audit_summary["fallback"]},
      "evidence_summary":[{"type":x.evidence_type.value,"source":x.source,"modality":x.modality,"representation":x.representation,"validation":x.validation_status,"confidence_source":x.confidence_source.value} for x in result.evidence],
      "limitations":([result.final_answer.text] if blocked else ["Confidence is UNKNOWN unless supplied by evidence."])}

def deterministic_report(result: AnalysisResult) -> dict[str, Any]:
    return {"query":result.provenance["query"],"plan":list(result.plan.steps),"answer":result.final_answer.text,"confidence":"UNKNOWN","provenance":dict(result.provenance),"trace":[{"name":x.name,"status":x.status.value} for x in result.trace.steps],**summarize(result)}
