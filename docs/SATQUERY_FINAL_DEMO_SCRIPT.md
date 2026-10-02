# SatQuery final 2–4 minute demo script

## Opening — about 20 seconds

“SatQuery AI is an agentic vision-language assistant for remote-sensing imagery. It does not send every request to one generic model. It reads the input configuration and question, selects a specialist, and returns the answer with provenance, warnings, and a public execution trace.”

## 1. Single-image VQA — about 30 seconds

1. Select **Single image**.
2. Choose the approved non-test S2 TRAIN patch `S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11`.
3. Select **Single-image VQA**.
4. Enter: “Is water visible in this image?”
5. Run the request.

Say: “The agent selects `SINGLE_IMAGE_VQA` and the S2 projector plus frozen Qwen specialist. This observed run returned `no`. The warning makes clear that the VQA evidence is internal validation, not a public benchmark claim.”

## 2. Optical + SAR — about 35 seconds

1. Select **Optical + SAR**.
2. Choose approved pair `S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22` with matching S1 `S1B_IW_GRDH_1SDV_20170717T064605_29UPV_35_22`.
3. Enter: “Use the SAR and optical information together. Is water visible?”
4. Run the request.

Say: “The route is `OPTICAL_SAR_ANALYSIS`. SatQuery uses the CROMA joint representation, the approved joint projector, and frozen Qwen. This observed run returned `Yes`. Our controlled comparison did not show a gain over the optical-only baseline, so we do not claim one.”

## 3. Bi-temporal change description — about 35 seconds

1. Select **Bi-temporal**.
2. Set T1/Before to LEVIR-CC validation `A/val_000001.png`.
3. Set T2/After to `B/val_000001.png`.
4. Enter: “What changed between these two images?”
5. Run the request.

Say: “A is the documented PRE image and B is POST. The agent selects Chg2Cap through `TEMPORAL_CHANGE_DESCRIPTION`. This verified run generated ‘a row of houses is built at the top of the scene.’ The route describes change in text and does not claim a change mask or area estimate.”

## 4. Scene description — about 30 seconds

1. Return to **Single image**.
2. Reuse the approved non-test S2 TRAIN patch.
3. Select **Remote-sensing scene description**.
4. Enter: “Describe this image.”
5. Run the request.

Say: “The agent selects `SINGLE_IMAGE_SCENE_DESCRIPTION` and the AdaptLLM remote-sensing Qwen2-VL specialist. The verified run generated ‘There are two green trees in the picture.’ This is functional integration evidence, not a benchmark score.”

## Close — about 20 seconds

Open the execution trace and provenance section, then download the JSON result.

Say: “Every supported result retains the selected route, input identity, specialist provenance, warnings, confidence policy, and an exportable public trace.”

## Optional safety demonstration — about 20 seconds

Ask: “Highlight the water body.”

Say: “Grounding is intentionally blocked because this release has no validated specialist or trustworthy pixel-coordinate mapping. SatQuery returns no fabricated box, mask, confidence, or fallback answer.”
