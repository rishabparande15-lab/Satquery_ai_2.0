# Pipeline 3 Exact 5,000-Area Scientific Re-validation

Date: 2026-09-15

## Current status

The former data blocker is resolved and the master revalidation is complete.
All 5,000 selected areas have valid S1, 12-band S2, reference maps, and metadata.
The exact 4,600/200/200 split was preserved. No samples were substituted,
excluded, or moved.

The frozen current-Pipeline-3 baseline was trained on 4,600 areas, selected on
200 validation areas, repeated at a predetermined second seed, and evaluated in
one sealed 200-area test event. The selected physical+joint-CROMA hybrid achieved
4.1034 pp test MAE, 9.5873 pp test RMSE, and 65.0% dominant-class accuracy.

Scientific verdict: **A — HEALTHY**. No P0/P1 defect or evidence-based repair
was found. The exact result, configuration, ablations, per-class analysis,
repeatability evidence, compute accounting, and limitations are documented in
`PIPELINE3_5000_TRAINING_REPORT.md`.

## Authoritative identities

- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`.
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
- Final model-state hash: `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634`.
- Final configuration hash: `32c8776f44a070fca265a8ae81d92fe4f22649b190af08f10fdff9eea2c9e80e`.
- Final scientific fingerprint: `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.

## Test discipline

Test feature rows were explicitly dropped from every baseline, analysis, and
repeat load. Model, seed, checkpoint, and selection rule were frozen first.
The final command then scored all frozen established ablations together and
wrote `final/test_receipt.json`; it refuses to run again while that receipt or
the final metrics exists. Independent metric recomputation matched exactly.

## Historical record

Earlier revisions of this document correctly blocked training when only 1,000
areas had local S2. Phase 6A.9 acquired the exact missing 4,000 training-area S2
patches and repeated the completeness audit. That historical blocked verdict is
superseded by the verified dataset fingerprint and completed experiment above.

The exact 5,000-area experiment is now the authoritative scientific checkpoint
for Pipeline 3's scene-level coverage task. Future experiments must be versioned
separately and must preserve this result and receipt.
