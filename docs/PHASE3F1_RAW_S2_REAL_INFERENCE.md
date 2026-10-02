# Phase 3F.1 — Raw S2 Fixture Acquisition + Real Adapter Execution Gate

Date: 2026-09-21  
Status: **PASS**

## Scope and safety boundary

This phase exercised the existing remote-sensing adapter with real raw
Sentinel-2 data. It did not modify the Pipeline-3 manifest, scientific split,
training code, model weights, scientific artifacts, or dataset receipts.

The Qwen model was used only as a language-facing smoke-test consumer. Its
output is not a scientific prediction, land-cover label, geospatial grounding
claim, or replacement for the established Pipeline-3 predictor.

## Authoritative source and fixture selection

No new imagery was downloaded for this gate. The fixture was selected from the
already validated external Pipeline-3 S2 root documented in
`docs/PIPELINE3_5000_DATA_ACQUISITION.md`:

```text
<machine-local-data-root>\comparison\raw-1000\BigEarthNet-S2
```

The acquisition provenance is the project-pinned BigEarthNet v2 LMDB route:

- source object: `BENv2.lmdb/data.mdb`
- source mirror: `hackelle/BigEarthNetV2-LMDB`
- immutable source revision:
  `118d1b6285c080ba8e4078414e1b8a243b18c9bd`
- source contract: exact Pipeline-3 area identity and 12-band S2 set
- container/provenance limitation: the project documentation identifies this
  as the pinned unofficial pre-conversion representation, not a claim that
  reconstructed TIFF container bytes equal the publisher's original TIFF
  bytes

Selection was deterministic: the first five manifest rows with
`validation.s2_valid == true`, split not equal to `test`, and all twelve band
files present under the validated root. The scientific split was only read,
never changed.

| Area ID | Split |
|---|---|
| `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_57_86` | train |
| `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_70_71` | validation |
| `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_89_58` | train |
| `S2A_MSIL2A_20170613T101031_N9999_R022_T34VER_56_07` | train |
| `S2A_MSIL2A_20170613T101031_N9999_R022_T34VER_69_35` | validation |

No synthetic image, arbitrary external imagery, or fabricated S2 tensor was
used for the real-data execution.

## Raw S2 validation

Every selected area contained exactly the canonical bands in this order:

```text
B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
```

All files were readable single-band finite GeoTIFFs with the expected native
dimensions and the established 10/20/60 m multiresolution contract:

```text
B01/B09: 20x20
B02/B03/B04/B08: 120x120
B05/B06/B07/B8A/B11/B12: 60x60
```

The existing repository preprocessing path was used: each band was read with
the repository's raster validation and reprojected with bilinear resampling
onto the B02 10 m 120x120 grid. Each resulting real cube was finite and had
shape `[12, 120, 120]`.

| Area suffix | Canonical min | Canonical max | Canonical mean |
|---|---:|---:|---:|
| `T33UUP_57_86` | 14.0 | 6812.0 | 1654.2646 |
| `T33UUP_70_71` | 63.0 | 6135.9375 | 1703.0542 |
| `T33UUP_89_58` | 40.0 | 6208.0 | 1834.2798 |
| `T34VER_56_07` | 95.0 | 465.0 | 176.6515 |
| `T34VER_69_35` | 35.0 | 6732.0 | 1401.9862 |

## Adapter candidates and selection

The existing adapter produced three deterministic language-facing candidates
from each real 12-band cube:

- `true_color`: B04/B03/B02, one 120x120 RGB image
- `false_color`: B08/B04/B03, one 120x120 RGB image
- `band_group_views`: three 120x120 RGB images covering all 12 bands in three
  documented groups

All candidates are explicitly information-losing pseudo-RGB views. No claim of
native Qwen multispectral support was made. `true_color` was selected for the
real inference gate because it is the lowest-context, one-image path already
validated by the earlier Qwen RGB execution gate.

The preprocessing fingerprints below are SHA-256 digests of the adapter
outputs, not scientific dataset fingerprints.

| Area suffix | True-color fingerprint | False-color fingerprint | Group-view fingerprint |
|---|---|---|---|
| `T33UUP_57_86` | `f762333a8e19c20aa91fe059b84c4c176378feecf74d3dc586e9355e2ecaf817` | `6b408fe3df2af879aa577ce940f736c6f253d9a5b0b3fb7bb586654a83a8075f` | `fe06db778a473a9ec44791ba6f12be2b84cd3c053bdd471ecfac8db358eb9af5` |
| `T33UUP_70_71` | `038eeb9610ebc058af4ab6f3add152caba98a622f44b6865c6f1686a945b8c10` | `1e99fbf4dcbf8d6bdda387902a9619990edd68b2c1fa4a6cb95320287bed084b` | `36d2f63afca0781fcb7111582bd5984e1a5321741e933be7dd8fc52a92f59a5a` |
| `T33UUP_89_58` | `98a73b4918d6cdd3d8544383fcd112f4f02d863246b7001e36aff5a5a9a14008` | `5296d146938aa9aca845f1cf23562163bf05d4a2297c35f192ff8e07213b1601` | `23bf8ec32fc4fbab209e9c56c41b7872ceae8ed39ee5f4b8160d1b04828a5113` |
| `T34VER_56_07` | `b799f58ddf2e0bb2c1de195e900fadef6fc26117cf2eb85b8f42748c2df4e78a` | `aa4b8bb5d54ea73fb886d752cd2286084f899b59b36ed13508d8f1373113ad6a` | `a4b34e91ec5876bf437245bf101e9c7a1e9aa591dc11b77b359c7d84ce8534a0` |
| `T34VER_69_35` | `31d2638401ce53ffe839e7469ab6931eb1b87b062f3e15d0f28229b905931b2a` | `91b96e0897d9ecd66c9d2b892f38410d9cb2f20a3f67d002ca497332751455f7` | `61770662519c72104f1de5780e6485d712a349e61a2738d99ec1746e0c5f2196` |

## Qwen CUDA execution

The existing adapter loaded the local official checkpoint
`Qwen/Qwen2.5-VL-3B-Instruct` in FP16 on CUDA and generated at most 16 tokens
per area with the fixed prompt:

> Describe the visible land-cover scene in one concise sentence. Do not infer
> labels, coordinates, or scientific measurements.

Checkpoint revision:
`66285546d2b821cf421d4f5eb2576359d3770cd3`

| Area suffix | Preprocess s | Generation s | Tokens | Output prefix |
|---|---:|---:|---:|---|
| `T33UUP_57_86` | 0.2166 | 18.5908 | 16 | “The image shows a mix of forested areas and open water bodies, with some” |
| `T33UUP_70_71` | 0.1260 | 2.0950 | 16 | “The image shows a mix of agricultural fields, forests, and open spaces with some” |
| `T33UUP_89_58` | 0.1353 | 2.3065 | 16 | “The image shows a mix of agricultural fields, including crops and pastures, inters” |
| `T34VER_56_07` | 0.1537 | 1.9664 | 16 | “The image shows a mix of green vegetation and some bare soil or rocky terrain.” |
| `T34VER_69_35` | 0.1586 | 2.3930 | 16 | “The image shows a mix of forested areas and open fields with some water bodies” |

Model load took 30.5012 s. The largest observed CUDA allocation was
7,554,314,240 bytes, with 7,656,701,952 bytes reserved. The largest reported
process RSS after inference was approximately 2.644 GiB. The first request is
the cold CUDA/generation request; subsequent timings are not directly
interchangeable with cold-start timing.

## Gate result

**PASS — REAL_S2_FIXTURE_ACQUIRED and REAL_S2_ADAPTER_INFERENCE_PASS**

- raw real S2 source: PASS
- exact 12-band validation: PASS
- canonical real cube construction: PASS
- deterministic adapter candidate diagnostics: PASS
- existing S2 adapter into Qwen FP16 CUDA: PASS
- scientific representation used: **false**
- grounding or scientific prediction claim: **not made**

The Phase 3F.1 gate is complete. The raw imagery remains external to Git, and
no scientific baseline or Pipeline-3 split was changed.
