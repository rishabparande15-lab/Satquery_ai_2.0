"""Build deterministic Pass 5E interpretation artifacts for sample 61_39."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.interpretation_adapter import ADAPTER_VERSION, interpret_evidence, technical_companion, validate_interpretation


SOURCE=ROOT / "artifacts" / "spatial_evidence" / "61_39" / "evidence.json"
OUTPUT=ROOT / "artifacts" / "interpretation" / "61_39"
QUESTION="What is present in this image?"


def canonical_hash(value):
    encoded=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    evidence=json.loads(SOURCE.read_text(encoding="utf-8"))
    simple=interpret_evidence(evidence,QUESTION,mode="simple")
    technical=technical_companion(evidence,QUESTION)
    validate_interpretation(simple); validate_interpretation(technical)
    if simple != interpret_evidence(evidence,QUESTION,mode="simple"):
        raise RuntimeError("simple interpretation is not deterministic")
    if technical != technical_companion(evidence,QUESTION):
        raise RuntimeError("technical interpretation is not deterministic")
    durations=[]
    for _ in range(200):
        started=time.perf_counter()
        interpret_evidence(evidence,QUESTION,mode="simple")
        durations.append((time.perf_counter()-started)*1000)
    answer={"sample_id":"61_39","question":QUESTION,"simple":simple,"technical":technical,
            "contract":"allowlisted deterministic templates over validated spatial evidence; no LLM/VLM/VQA"}
    provenance={"adapter_version":ADAPTER_VERSION,"source_evidence":str(SOURCE.relative_to(ROOT)).replace("\\","/"),
        "source_evidence_sha256":canonical_hash(evidence),"simple_semantic_sha256":simple["semantic_sha256"],
        "technical_semantic_sha256":technical["semantic_sha256"],"determinism_check":"PASS; two exact structured outputs matched per mode",
        "benchmark":{"iterations":len(durations),"median_milliseconds":statistics.median(durations),
                     "p95_milliseconds":sorted(durations)[int(len(durations)*.95)-1]},
        "limitations":["Evidence strength is descriptive and uncalibrated.","Regions are deterministic feature-threshold groups, not learned segmentation.",
                       "Optical and SAR claims are complementary sensor views, not a joint semantic prediction."]}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT / "answer.json").write_text(json.dumps(answer,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (OUTPUT / "simple_answer.txt").write_text(simple["answer"]+"\n",encoding="utf-8")
    (OUTPUT / "technical_answer.txt").write_text(technical["answer"]+"\n",encoding="utf-8")
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(OUTPUT),"status":simple["status"],"benchmark":provenance["benchmark"]},indent=2))


if __name__ == "__main__":
    main()
