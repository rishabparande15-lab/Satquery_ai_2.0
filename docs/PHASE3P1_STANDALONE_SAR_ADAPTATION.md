# Phase 3P.1: Standalone SAR adaptation pilot

Status: `PHASE3P1_COMPLETE`.

This was a bounded, standalone Sentinel-1 pilot. It did not load an S2 input,
perform optical-SAR fusion, alter the validated Sentinel-2 projector, or modify
controller capability gates.

## Data and integrity

The authoritative Phase 3O.12 TRAIN/VALIDATION manifest was joined only through
`raw-1000/metadata.parquet`'s exact `patch_id -> s1_name` relation. Valid joined
pools were Binary 23/50, MCQ 24/50, and Caption 26/50 (TRAIN/VALIDATION). The
preregistered balanced panel selected 6/3 records per task: 18 TRAIN and 9
VALIDATION. All 26 distinct S1 samples passed the existing VV/VH, 10 m,
finite-value, and robust-normalization path. There were no duplicate record
IDs, no duplicate image/task identities, no missing S1 samples, no
train-validation image overlap, and TEST access was zero.

## Training and provenance

Only `S1SARProjector` was optimized: 5,772,800 trainable parameters with
AdamW, learning rate 0.0005, batch size 1, one epoch, no accumulation, and 18
maximum optimizer steps. CROMA and Qwen were frozen; the S2 projector was not
loaded and had zero optimizer/training membership.

Initial projector SHA-256:
`2329af637b08061a48bf7e317ee9e8511299710bed1531280b4e953036d243dd`.
Initial fingerprint:
`90bccf296ee77f82ede40ded781e13312c61ab27ad5fa98963d26fb7d7f55ac9`.
Final projector SHA-256:
`a7e57bff0cba141db5c34f549d0bc3f25638960fe39ab25246bbab49e979accd`.
Final fingerprint:
`4bbd8bffdcd2b1bb66851c381eb0e1ed6f3c21b7486250f02421426ffe0505dc`.

No-step diagnostics had finite losses, supervised-token counts 2/2/33, and
six nonzero projector gradient tensors per task. Qwen, CROMA, and S2-projector
gradient tensors were all zero. Training loss was 6.3985 at step 1, 3.7592 at
the final step, and 4.8726 on average.

## Validation, reload, and semantic behavior

Final validation mean loss was 3.3182932469579907 (Binary 4.6602923, MCQ
1.7070021, Caption 3.5875854), all finite. A fresh process reloaded CROMA,
Qwen, the final SAR projector, and the exact validation panel. The mean matched
exactly: absolute difference 0.0 (`EXACT_MATCH`; absolute tolerance 1e-6).

Before training, all generated responses were empty: Binary and MCQ accuracy
were 0/3 with all responses unparsable; captions were empty 3/3. After training,
Binary and MCQ accuracy were still 0/3 under correct, shuffled, and zero SAR.
Correct-SAR outputs changed versus shuffled SAR on 2/9 records and versus zero
SAR on 7/9. Captions became nonempty 3/3 for correct/shuffled SAR but remained
empty for zero SAR. The mean teacher-forced correct-minus-shuffled target-score
margin was -0.0268, which is not evidence of a correct-image advantage.

## Conclusion

Classification: `SAR_TRAINING_PRODUCES_NO_CLEAR_GAIN`.

The protocol and reload are valid, and Qwen, CROMA, and the S2 projector were
unchanged. Loss reduction, caption text emergence, and zero-SAR sensitivity do
not demonstrate image-conditioned semantic reasoning. Standalone SAR is **not
eligible for controlled integration**. SAR routing and S1+S2 fusion remain
blocked.

Exact recommended next phase: `Phase 3P.2 — standalone SAR supervision and
generation-interface diagnosis`, restricted to answer-token priors,
caption-prefix collapse, and correct-vs-shuffled target-score behavior on this
fixed non-TEST panel. It must not begin S1+S2 fusion or unblock controller
routing.
