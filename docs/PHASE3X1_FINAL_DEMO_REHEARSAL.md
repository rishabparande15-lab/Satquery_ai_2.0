# Phase 3X.1 — final SIH demo rehearsal and release freeze

Status: `PHASE3X1_DEMO_READY`
Final SIH demo status: `READY_WITH_LIMITATIONS`

The four required workflows were rehearsed in a real Chrome browser through the unified UI, `POST /api/v1/query`, `SatQueryAgent`, their assigned specialists, normalized result rendering, public provenance/warnings/trace, and JSON export. Each main happy path had zero console errors, page errors, and failed network requests. The optional grounding request stayed explicitly blocked.

## Frozen workflows

| Demo | Approved non-test input | Route | Specialist | Rehearsal outcome |
| --- | --- | --- | --- | --- |
| Single-image VQA | BigEarthNet TRAIN `S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11` | `SINGLE_IMAGE_VQA` | S2 projector + frozen Qwen2.5-VL | Completed; generated `no` |
| Optical + SAR | Approved fixed non-test validation pair `S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22` + matching S1 | `OPTICAL_SAR_ANALYSIS` | CROMA joint + projector + frozen Qwen2.5-VL | Completed; generated `Yes` |
| Bi-temporal change | LEVIR-CC validation `val_000001.png`, A/PRE → B/POST | `TEMPORAL_CHANGE_DESCRIPTION` | Chg2Cap | Completed; nonempty change description |
| Scene description | Same approved BigEarthNet TRAIN patch, explicit B04/B03/B02 RGB rendering | `SINGLE_IMAGE_SCENE_DESCRIPTION` | AdaptLLM remote-sensing Qwen2-VL-2B | Completed; nonempty scene description |

## Safety boundary

`SINGLE_IMAGE_GROUNDING` was deliberately rehearsed as `BLOCKED`. It returned no fabricated bounding box, mask, confidence, or VQA fallback.

## Readiness conditions verified

- Public operational traces contain input configuration, query classification, route selection, validation/policy, specialist execution where available, and result integration. They contain no private reasoning.
- Response provenance identified the selected input(s) and route-specific specialist material. The temporal record states PRE/POST; the optical/SAR record contains the paired S1 and S2 inputs and CROMA-joint specialist provenance.
- The non-calibrated routes return `confidence.value = null` and `confidence.type = NOT_AVAILABLE`.
- All four primary flows exported `satquery-agent-result.json` after rendering. Receipts check the response contract and timestamped execution trace rather than exposing hidden reasoning.
- Browser loading recovered after each request; duplicate submissions are disabled while an analysis is running.
- No test content, labels, inference, or metrics were used in Phase 3X.1.

## Scientific boundaries preserved

This rehearsal does not change the Phase 3X compliance evidence: 17 functional criteria comprise **4 COMPLETE**, **8 COMPLETE_WITH_LIMITATIONS**, **4 PARTIAL**, and **1 BLOCKED** (12/17 complete or complete-with-limitations). It makes no SOTA, benchmark-superiority, calibrated-confidence, spatial-evidence, or Cartosat/RISAT sensor-generalization claim.

The scene route's runtime warnings correctly state that it uses an explicit RGB rendering and that confidence is uncalibrated. Its missing benchmark evaluation is retained as a release limitation, not inferred away from a successful functional demonstration.

## Release freeze

Feature development stops here. The next work, only with explicit authorization, is **final SIH submission / presentation packaging**: presentation, final report, architecture diagram, setup instructions, viva preparation, judge-facing limitations, repository cleanup, and an optional release commit/tag.

The only remaining external dependencies are authorized VRSBench/RSVQA evaluation data, CDVQA author clarification for the separate Change-VQA research track, and validated Cartosat-2S/RISAT data plus adapters.
