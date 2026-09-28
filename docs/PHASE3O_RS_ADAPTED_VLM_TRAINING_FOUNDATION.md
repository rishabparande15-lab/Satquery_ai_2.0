# Phase 3O — RS-Adapted VLM Training Foundation

Status: **FOUNDATION PASS; ADAPTATION PILOT BLOCKED BEFORE TRAINING**

## Outcome

Phase 3O adds a typed, executable training-data boundary for a frozen-Qwen,
learned-S2-projector adaptation pilot. It does not run a new model update,
create a checkpoint, or report a benchmark result. The only accessible
canonical image-language asset is the validated BigEarthNet.txt view; no
checksum-resolved local raw S2 raster root is available in this checkout.

The prior Phase 3G pilot is not reused as a Phase 3O result: its selection
includes spatial-question records, while this phase accepts only the canonical
`VQA_BINARY`, `VQA_MULTIPLE_CHOICE`, and `CAPTIONING` task types.

## Training contract

`src/eo_vlm/adaptation.py` introduces `TrainingRecord`. Every record requires
sample/image identity, inherited split, exact language supervision, source
annotation revision, representation fingerprint, raw-source provenance,
license state, and augmentation policy. It rejects test records, grounding,
temporal/change tasks, unknown licenses, undeclared augmentation, scientific
representations, unsupported input modes, and train/validation image overlap.

Two explicit modes exist:

- `MODE_RGB`: resolved RGB image through the Qwen RGB processor.
- `MODE_S2_PROJECTED`: canonical raw `[12,120,120]` S2 in exact
  `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12` order into the learned
  `[1,16,2048]` Qwen-token projector.

`hybrid_830d`, CROMA, SAR, optical-SAR fusion, temporal pairs, and grounding
are excluded. Qwen remains frozen; only the 5,549,184-parameter S2 projector
is a permitted trainable component. The Phase 3G measured resource envelope
requires FP16 Qwen, batch one, gradient accumulation eight, and no full-model
copy. PEFT and bitsandbytes are unavailable in this environment.

## Data and execution gate

The canonical asset has 101,542 validated records: 93,208 train, 4,175
validation, and 4,159 test. Phase 3O asks for 200 train and 75 validation
records, selected after verified local raw-asset resolution in canonical
source order. Test is never read. This checkout resolves zero eligible raw S2
assets: the historical Phase 3G source root was external to this workspace.

Accordingly, `artifacts/training/phase3o/` records the configuration, zero
selected records, blocked run receipt, and absent checkpoint manifest. No
loss, gradient, parameter change, validation, or checkpoint reload is claimed.

## Validation

Focused tests validate record traceability, RGB/S2 mode boundaries, safe task
allowlisting, caption supervision, test rejection, scientific-input rejection,
license/augmentation rejection, leakage detection, local-asset gating, and
checkpoint manifest behavior. The adaptation validation artifact records the
same decision.

## Next gate

Supply an authorized local S2 root with each selected image's twelve files and
checksum evidence. Then the provided record selector can produce the 200/75
manifest and run a projector-only `ADAPTATION_PILOT`; acceptance requires
finite loss, observed projector parameter change, validation loss, an
adapter-only checkpoint hash, and reload verification.

The locked scientific baseline remains 65.0% accuracy, 4.1034286734 pp MAE,
and 9.5873312123 pp RMSE, unchanged.
