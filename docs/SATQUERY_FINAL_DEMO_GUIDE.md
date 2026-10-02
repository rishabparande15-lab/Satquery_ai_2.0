# SatQuery final SIH demo guide

Use only the approved non-test inputs below. Every workflow goes through the unified UI and `POST /api/v1/query`; wait for the result and use its JSON export where appropriate. Do not claim a benchmark result, calibrated confidence, spatial grounding, or sensor-domain generalization.

## Recommended two-minute sequence

1. **Single-image VQA.** Choose **Single image**, select `S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11` (TRAIN), select **Single-image VQA**, and ask: “Is water visible in this image?” SatQuery routes to `SINGLE_IMAGE_VQA` and renders a generated answer, the specialist, warnings, provenance, and trace. Tell judges: “This is a real S2-to-language proof of concept with an explicit non-benchmark limitation.”

2. **Optical + SAR.** Choose **Optical + SAR**, select approved pair `S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22` with its matching S1 input, and ask: “Use the SAR and optical information together. Is water visible?” The route is `OPTICAL_SAR_ANALYSIS`. Tell judges: “Both modalities are sent to the joint CROMA route; the controlled internal comparison did not show a gain over S2-only.”

3. **Bi-temporal change description.** Choose **Bi-temporal**, set T1/Before to `val_000001.png` from `A` and T2/After to the same filename from `B`, then ask: “What changed between these two images?” The route is `TEMPORAL_CHANGE_DESCRIPTION`. Tell judges: “A is authoritative PRE and B is POST; Chg2Cap generates a natural-language change description, not a mask or area estimate.”

4. **Scene description.** Choose **Single image**, reuse the approved BigEarthNet TRAIN patch, select **Remote-sensing scene description**, and ask: “Describe this image.” The route is `SINGLE_IMAGE_SCENE_DESCRIPTION`. Tell judges: “The model produces a real description from an explicit RGB rendering; this integration has no benchmark evaluation receipt.”

5. **Optional safety boundary.** In **Single image**, select **Grounding** and ask: “Highlight the water body.” `SINGLE_IMAGE_GROUNDING` must remain `BLOCKED`. Tell judges: “SatQuery refuses to fabricate a box, mask, confidence, or fallback answer when no validated grounding specialist is available.”

## Presenter checks

For each primary flow confirm that the route, specialist, public execution trace, provenance, warnings, and JSON export are visible; confidence remains **Not calibrated / unavailable** where shown. Let a prior model remain cached during a live presentation only when it does not change the scientific behavior or selected model.
