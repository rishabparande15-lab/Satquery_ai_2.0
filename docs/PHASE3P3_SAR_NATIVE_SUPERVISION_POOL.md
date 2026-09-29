# Phase 3P.3: SAR-native supervision pool construction

Status: `PHASE3P3_PARTIALLY_COMPLETE`.

This phase constructed and audited a new candidate pool only. It did not train
the SAR projector, create an optimizer, load Qwen, enable controller routing,
or perform S1+S2 fusion. TEST rows were excluded before candidate evaluation.

## Source and taxonomy

The non-TEST `raw-1000/metadata.parquet` source supplied 800 records from its
authoritative TRAIN and VALIDATION partitions. Each has a paired BigEarthNet-S1
identity and CLC-derived multi-label metadata. BigEarthNet describes its S1
patches as paired with S2 patches and its labels as derived from CLC2018; the
CLC nomenclature defines inland and marine water bodies. Sentinel-1 supports
VV/VH acquisition, and published backscatter work establishes water as a
conservative SAR-discriminable broad class. See [BigEarthNet](https://bigearth.net/),
[CLC nomenclature](https://land.copernicus.eu/content/corine-land-cover-nomenclature-guidelines/html/),
[ESA Sentinel-1 products](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Data_products),
and [Sentinel-1 backscatter signatures](https://doi.org/10.1038/s41597-021-01059-7).

Included task: a single balanced binary question asking whether the patch has
an authoritative CLC inland- or marine-water label. Rejected task categories:
country/climate/season, exact area/topology/adjacency, fine optical or species
semantics, a forced four-option broad-class MCQ, and free-form captioning.
Those have no authoritative, mutually exclusive SAR-native target in this
source and were not fabricated.

## Pool and audit

The pool contains 100 records: 80 TRAIN and 20 VALIDATION, exactly balanced
(50 yes / 50 no overall; 40/40 train; 10/10 validation). Every candidate has a
verified S1 asset, validated preprocessing, a VV/VH statistic summary, a CROMA
representation summary, full source labels and citations, and an opposite-label
shuffled pairing. There is no train-validation patch overlap and TEST access is
zero.

As a non-decisive sanity check, water-label candidates averaged VV −16.22 dB
and VH −24.12 dB, versus non-water VV −10.62 dB and VH −16.78 dB. This confirms
the selected groups are not obviously identical; it does not create labels or
validate a trained model.

## Classification

`SAR_NATIVE_POOL_PARTIALLY_READY`: water/non-water is a conservative,
provenanced SAR-supportable candidate task, but the source still supplies
CLC-derived labels rather than independently audited SAR-native targets and
does not provide defensible MCQ or description supervision. Therefore bounded
SAR retraining is **not yet justified**. The required next phase is independent
SAR-native reference/human provenance audit of the proposed water candidates,
followed by a decision gate before any training. SAR routing and fusion remain
blocked.
