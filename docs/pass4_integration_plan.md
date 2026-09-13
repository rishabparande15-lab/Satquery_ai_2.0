# Pass 4 Integration Plan

## Findings

The most valuable usable input is the selected 5,000-area BigEarthNet v2 archive. It adds 4,000 areas beyond the current 1,000-area baseline and supplies a complete SAR-only branch plus reference maps and labels. It does not add VQA, captioning, learned grounding, or genuine temporal supervision. The official archive should remain blocked until all stream parts and metadata are available.

## Task value assessment

| Capability | Result | Evidence |
|---|---|---|
| CROMA representation | YES for SAR; POSSIBLY for joint | Complete VV/VH TIFFs are present; S2 rasters are absent |
| Land-cover prediction | YES | 5,000 metadata rows with 19 multi-label classes and 5,000 reference maps |
| VQA | NO | No questions or answers |
| Captioning | NO | No caption text or image-caption pairs |
| Learned grounding | NO as a learned task; POSSIBLY for deterministic map alignment | Reference maps exist, but no referring expressions or boxes |
| Temporal/change prediction | NO | No genuine before/after pairs or change targets |
| Optical-SAR fusion | POSSIBLY | S1 is complete, but matching S2 rasters are missing from the selected archive |
| Confidence calibration | POSSIBLY for land-cover after a leakage-safe validation experiment | Labels and a validation split exist; no calibrated prediction set is supplied |
| Final domain evaluation | YES for SAR/land-cover; NO for VQA/caption/temporal | Tasks and targets are present only for land-cover/SAR |

## Recommendation

1. Integrate the selected archive first as an additive SAR/reference-map evaluation source.
2. Consume it through `src/dataset_loader.py` after an archive-aware adapter materializes the existing expected `BigEarthNet-S1` and `Reference_Maps` layout. Keep the current Pass 3 manifest and metrics unchanged.
3. Use the existing SAR CROMA encoder directly for the two float32 120x120 SAR bands. Joint CROMA requires the matching S2 rasters, which are absent from this ZIP, so it is adaptable rather than direct.
4. Use multi-label average precision/F1 and the repository's existing land-cover metrics for classification; use pixel-map metrics only after confirming the reference-map class coding against the existing loader.
5. Build a new geographic-area split excluding all 1,000 Pass 3 patch IDs before claiming generalization.

## Required preprocessing

Validate VV/VH pairing by area ID, preserve UTM CRS and 10 m transform, apply the existing SAR normalization profile, map metadata labels through the existing 19-class vocabulary, and retain source ZIP hashes in provenance. Do not silently substitute the S2 patch ID for the S1 acquisition identity.

## Keep separate

The 600/200/200 Pass 3 experiment, its metrics, provenance, leakage audit, and CROMA implementation remain unchanged. Any expanded 5,000-area experiment must receive a new manifest, split file, artifact root, and metric namespace.

## Blockers and risks

The selected archive lacks S2 imagery, so it cannot independently validate optical or joint CROMA. It overlaps the entire existing baseline. The official fragment archive has no usable sample records. No task-specific VQA, captioning, grounding, temporal, or calibration target is present.

## Suggested next benchmark

After leakage-safe split construction, run a small SAR-only representation/load benchmark first, then a separately named expanded evaluation. Do not train large models in Pass 4.