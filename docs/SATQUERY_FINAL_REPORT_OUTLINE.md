# SatQuery final report outline

## 1. Abstract

State that SatQuery is a local agentic remote-sensing assistant. Summarize the four verified routes, unified provenance-first response contract, and the limits on benchmark and sensor-generalization claims.

## 2. Introduction

Explain the need to handle single imagery, optical-SAR pairs, and temporal pairs without collapsing their data contracts into one generic request.

## 3. Problem statement

Describe the SIH PS-26167 scope and the need for useful, traceable remote-sensing interaction.

## 4. Objectives

- Route supported input configurations to an appropriate specialist.
- Provide useful natural-language answers or descriptions with provenance and warnings.
- Fail closed for unsupported or unvalidated capabilities.

## 5. Related work

Cover remote-sensing VQA, optical-SAR representations, change captioning, and remote-sensing vision-language models. Distinguish cited work from SatQuery's own verified results.

## 6. System architecture

Document the unified UI, `POST /api/v1/query`, `SatQueryAgent`, input configuration analysis, query intent classification, capability registry, specialists, response normalization, provenance, warning policy, confidence policy, execution trace, and export.

## 7. Datasets and data boundaries

Describe BigEarthNet development inputs, admitted LEVIR-CC development pair policy, and the no-test-access policy. State that CDVQA remains a separate blocked provenance track.

## 8. Preprocessing and input validation

Describe canonical S2 and S1 input expectations, role-specific upload checks, PRE/POST chronology requirements, RGB rendering for scene description, and fail-closed error handling.

## 9. Models and specialists

Describe the S2 projector plus frozen Qwen2.5-VL, CROMA joint plus projector plus frozen Qwen2.5-VL, Chg2Cap, and AdaptLLM remote-sensing Qwen2-VL-2B. Do not claim that they form one trained model.

## 10. Agentic controller

Explain how declared roles and query intent select one route. Include blocked grounding as a deliberate safety policy.

## 11. Single-image VQA

Present the approved input contract, internal binary and MCQ measurements, and the absence of public benchmark evaluation.

## 12. Optical-SAR integration

Present matched S1/S2 input handling, CROMA joint representation, result contract, and the finding that no gain over the optical-only baseline was demonstrated.

## 13. Bi-temporal change analysis

Describe LEVIR-CC A/PRE and B/POST handling, Chg2Cap inference, and the text-only boundary without a mask or area estimate.

## 14. Scene description

Describe the explicit RGB input/rendering path, the remote-sensing Qwen specialist, and functional verification without a benchmark evaluation receipt.

## 15. Frontend and API

Show the three input modes, browser-visible route/provenance/trace, export, loading behavior, and structured errors.

## 16. Evaluation and results

Use [SATQUERY_FINAL_RESULTS.md](SATQUERY_FINAL_RESULTS.md). Keep every metric tied to its own experiment or functional check; do not derive a combined accuracy.

## 17. Scientific limitations

Include the exact limits in [SATQUERY_SCIENTIFIC_LIMITATIONS.md](SATQUERY_SCIENTIFIC_LIMITATIONS.md): internal VQA validation, no fusion gain, text-only temporal output, no scene benchmark receipt, blocked grounding, unvalidated Cartosat/RISAT transfer, and uncalibrated confidence.

## 18. SIH requirement compliance

Include the unchanged evidence counts: 17 criteria, 4 complete, 8 complete-with-limitations, 4 partial, and 1 blocked. Reference [SIH_26167_COMPLIANCE_MATRIX.md](SIH_26167_COMPLIANCE_MATRIX.md).

## 19. Future work

Limit proposals to authorized VRSBench and RSVQA evaluation, validated grounding, spatial change masks, calibrated uncertainty, better multimodal fusion, and validated Cartosat-2S/RISAT adapters.

## 20. Conclusion

Summarize the verified routes and safety boundaries without claiming benchmark leadership or unsupported capabilities.

## 21. References

List the official papers, repositories, and dataset sources for CROMA, Qwen2.5-VL, Chg2Cap, LEVIR-CC, AdaptLLM remote-sensing Qwen2-VL, BigEarthNet, and any SIH problem statement material used in the report.
