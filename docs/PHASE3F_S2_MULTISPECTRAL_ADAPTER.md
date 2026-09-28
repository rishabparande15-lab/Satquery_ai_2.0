# PHASE 3F — Real S2 Multispectral Input Adapter and Qwen Smoke Test

**Status:** deterministic S2 adapter `IMPLEMENTED`; local Qwen processor
compatibility `TESTED` on a controlled fixture; real S2 inference `BLOCKED`
because no raw 12-band S2 imagery is present in the repository checkout.

## 1. Objective

Establish a reproducible language-facing path from the canonical S2 tensor to
Qwen's existing RGB image processor without changing scientific preprocessing
or claiming native multispectral understanding.

The implemented result means:

> Qwen can receive a deterministic S2-derived representation through the
> language-facing adapter.

It does not mean that Qwen understands all 12 Sentinel-2 bands, has learned
multispectral semantics, or has native multispectral model support.

## 2. S2 Input Contract

The adapter accepts one sample as `[12,120,120]` or `[1,12,120,120]` and requires
this exact canonical order:

```text
B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
```

Batch sizes greater than one are rejected for the single-request Qwen path.
Band order, channel count, spatial size, finite values, and declared nodata
handling are validated before conversion. The original S2 reference and
canonical band order are retained in provenance.

## 3. Existing Scientific Preprocessing

The adapter reuses the repository's existing per-channel robust scaling
convention from `src/preprocessing.py`: mean ± two standard deviations,
clipped to `[0,1]`, with non-finite/nodata values represented as zero. No
scientific tensor is modified. The language-facing conversion is a separate
`LANGUAGE_FACING_S2_ADAPTER` stage after validation.

## 4. Language-Facing Adapter Boundary

Implemented in `src/eo_vlm_adapter.py` as `prepare_s2(...)`. It produces one or
three deterministic H×W×3 uint8 views that Qwen's existing processor can
consume. The adapter records strategy, bands used, output shapes, normalization,
information loss, source reference, and a preprocessing fingerprint.

## 5. Candidate Representations

| Strategy | Mapping | Output | Information loss |
| --- | --- | --- | --- |
| `true_color` | `B04/B03/B02` → RGB | one `[120,120,3]` uint8 view | `true`; 9 bands not represented |
| `false_color` | `B08/B04/B03` → RGB | one `[120,120,3]` uint8 view | `true`; 9 bands not represented |
| `band_group_views` | groups `(B01..B04)`, `(B05..B08)`, `(B8A,B09,B11,B12)`; each group maps channels 1, 2, mean(3,4) to RGB | three `[120,120,3]` uint8 views | `true`; each fourth-band pair is combined and Qwen semantics are not native |

The grouped representation is an information-preserving *input coverage
experiment* only in the sense that every canonical band contributes to a view;
it is not lossless and does not create learned multispectral fusion. Its three
views can be passed as multiple images only through an explicitly supported
Qwen multi-image path; that path was not used for real inference here.

## 6. Normalization

Normalization is deterministic and documented:

```text
per-channel mean ± 2 std → clip to [0,1] → uint8 [0,255]
```

Declared nodata values are converted to the adapter's zero representation.
Unexpected NaN/Inf values are rejected. No random projection, PCA, learned
projection, or dataset-fitted transform was introduced.

## 7. Information Loss

All three strategies set `information_loss = true`.

- True-color and false-color select only three of twelve bands.
- Grouped views use all bands, but each four-band group is reduced to three
  display channels using a deterministic mean for its third/fourth pair.
- Qwen receives RGB-like images and has no native band identity metadata in its
  visual encoder.

Therefore the adapter provides S2-compatible visualization/input routing, not
native or learned multispectral understanding.

## 8. Selected Smoke-Test Representation

No real S2 sample was available locally: the repository's `data/raw` tree
contains metadata/source artifacts but no 12-band raster or tensor file, and
no new dataset was downloaded. Consequently no representation was selected for
a real Qwen S2 inference run.

For a future real-data smoke test, `true_color` is the baseline candidate
because it is one image, directly compatible with the already validated Qwen
RGB processor, deterministic, and computationally least expensive. This is a
technical compatibility choice, not a scientific superiority claim.

## 9. Qwen Processor Integration

Processor compatibility `TESTED` on an in-memory deterministic 12-band
fixture. The true-color output was accepted by the local Qwen processor and
produced:

```text
input_ids, attention_mask, pixel_values, image_grid_thw
pixel_values shape: (64, 1176)
```

This confirms adapter-to-processor compatibility only. It is not a real S2
model inference result.

## 10. Qwen Inference Results

Real S2 Qwen inference: `BLOCKED` — no local raw 12-band S2 sample exists.

No synthetic fixture was passed to the 7.5 GB CUDA model as a substitute for
real S2 imagery. Therefore there is no honest S2 generation text, S2
generation latency, or S2 peak VRAM measurement to report.

The previously validated RGB Qwen path remains unchanged and operational.

## 11. GPU/RAM Measurements

No S2 inference resource measurements were taken. The existing Qwen FP16
reference remains approximately 7.52 GB allocated after model load on the
8,151 MiB GPU, leaving limited headroom. Batch size one and small generation
limits remain mandatory for the next real-data test.

## 12. Provenance

Each prepared S2 result preserves:

- source image ID and source S2 reference;
- canonical band order and bands used;
- adapter ID/version;
- normalization strategy;
- representation strategy;
- output dimensions and dtype;
- preprocessing SHA-256 fingerprint;
- Qwen model ID and checkpoint revision;
- explicit `information_loss = true`;
- explicit `scientific_representation_used = false`.

## 13. Limitations

- No real S2 tensor was available for the required end-to-end generation.
- The RGB-oriented Qwen vision encoder does not receive explicit spectral-band
  semantics through these views.
- Grouped views are deterministic display adaptations, not learned fusion.
- SAR, optical-SAR, temporal, grounding, CROMA projection, and hybrid input
  remain outside this phase.
- No VQA, captioning, or scientific metric was computed.

## 14. Implemented vs Future Multispectral Learning

| Capability | Status |
| --- | --- |
| Exact 12-band validation | `IMPLEMENTED / TESTED` |
| Deterministic true-color view | `IMPLEMENTED / TESTED` |
| Deterministic false-color view | `IMPLEMENTED / TESTED` |
| Deterministic grouped views | `IMPLEMENTED / TESTED` |
| Qwen processor acceptance | `TESTED` on controlled fixture |
| Real S2 → Qwen generation | `BLOCKED` by missing local S2 imagery |
| Native multispectral Qwen support | `UNSUPPORTED / NOT CLAIMED` |
| Learned multispectral representation | `DEFERRED` |
| Projector training | `DEFERRED` |

## 15. Phase 3G Readiness

Phase 3G is ready only after obtaining an authorized existing project S2
sample/tensor. The next run should route at most five real samples through
`prepare_s2(strategy="true_color")`, the Qwen processor, and the already
validated FP16 CUDA model, recording preprocessing, processor, generation,
VRAM, RAM, and provenance measurements.

Focused tests: **38 passed**. Compile checks and `git diff --check` passed.
The scientific baseline remains 65.0% accuracy, 4.1034286734 pp MAE, and
9.5873312123 pp RMSE. No CROMA, scientific predictor, dataset split,
representation, receipt, checkpoint, or Phase 2F semantic artifact was
modified.
