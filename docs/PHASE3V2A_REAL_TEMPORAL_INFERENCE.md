# Phase 3V.2A — real temporal inference unblock

Status: `PHASE3V2A_MODEL_INFERENCE_COMPLETE_UI_PENDING`.

The official LEVIR-CC repository directly links the Hugging Face
`lcybuaa/LEVIR-CC` distribution. The distribution revision used was `881887b`;
its archive SHA-256 is
`e05d38c0fdfda8c9b2048d314e5f95974d8b81e1b9f83f107acc39d55015e130` and its
dataset-card metadata declares Apache-2.0. This records the actual linked
distribution only; it does not establish every upstream imagery license.

The ZIP was inspected only through directory metadata. Exactly five
deterministic validation pairs were extracted: `val_000001.png` through
`val_000005.png`, both A/PRE and B/POST copies. All are RGB 256×256 pairs.
No caption JSON or TEST image was extracted or opened.

`TEST_CONTENT_ACCESS = 0`, `TEST_LABEL_ACCESS = 0`, `TEST_INFERENCE = 0`, and
`TEST_METRICS = 0`.

The author-controlled RSCaMa `data/LEVIR_CC/vocab.json` is compatible with the
official Chg2Cap checkpoint: 501 tokens, output dimension 501, special IDs
`<NULL>=0`, `<UNK>=1`, `<START>=2`, and `<END>=3`. Its published preprocessor
uses threshold five and sorted token-index construction, matching Chg2Cap.

Real CUDA Chg2Cap inference generated, for validation pair `val_000001.png`:

> a row of houses is built at the top of the scene

The direct run and the real local HTTP API run both completed successfully.
Five validation smoke runs were nonempty. Reversing T1/T2 changed the output;
T1/T1 and T2/T2 were rejected by the intentional identical-input safeguard.
No benchmark metrics are claimed.

The BI-TEMPORAL UI exists, but a real Playwright run remains pending because the
workspace lacks a usable Playwright browser runtime. No mocked browser success
path was used. The next phase is `PHASE3V2B_REAL_BROWSER_VERIFICATION`.
