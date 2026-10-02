# SIH PS-26167 compliance matrix

| Requirement | Implementation | Status | Limitation |
| --- | --- | --- | --- |
| Single-image VQA | S2 projector + frozen Qwen | COMPLETE_WITH_LIMITATIONS | Internal validation only. |
| Second single-image task | AdaptLLM remote-sensing scene description | COMPLETE_WITH_LIMITATIONS | RGB rendering/upload only; no benchmark receipt. |
| Optical + SAR | CROMA joint + Qwen | COMPLETE_WITH_LIMITATIONS | No demonstrated gain over S2-only control. |
| Bi-temporal change analysis | Chg2Cap PRE/POST description | COMPLETE_WITH_LIMITATIONS | Text only; no change mask/area. |
| Agentic orchestration/UI/trace | SatQueryAgent + unified API + browser | COMPLETE | Fail-closed routing. |
| Input support | Role-specific upload validators | PARTIAL | Not a universal arbitrary-raster reader. |
| VRSBench/RSVQA | Interface/task readiness | PARTIAL | No authorized benchmark evaluation executed. |
| CDVQA | Preserved research provenance | BLOCKED | Awaiting author confirmation; not required for change description. |
| Cartosat-2S/RISAT | Adapter specification | PARTIAL | Sensor-domain transfer unvalidated. |

The functional SIH routes are demonstrably usable, but this is not a benchmark, SOTA, or sensor-generalization claim.
