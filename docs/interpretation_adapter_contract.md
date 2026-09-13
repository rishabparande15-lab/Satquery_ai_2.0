# Pass 5E Interpretation Adapter Contract

The adapter transforms validated `spatial_evidence_v1` claims into controlled text. It does not inspect imagery, raw feature values, CROMA dimensions, or user wording to invent a semantic result. Output status is `ANSWERED` only when an allowlisted claim passes its exact claim text, epistemic type, sensor, feature, confidence-status, region-reference, and source-artifact checks.

| Claim ID | Required source | Meaning | Allowed wording | Forbidden wording | Spatial requirement | Status wording |
|---|---|---|---|---|---|---|
| `vegetation` | Optical `NDVI`, exact evidence-schema claim | Derived vegetation-related spectral evidence | “derived optical evidence … vegetation-related spectral evidence” | forest, crop species, land-cover identity | Location only from referenced region token geometry | `WEAK`, `MODERATE`, or `STRONG` evidence; uncalibrated |
| `water` | Optical `NDWI`, exact evidence-schema claim | Derived water-related spectral evidence | “derived optical evidence … water-related spectral evidence” | river, lake identity, boundary, area | Same as above | Same as above |
| `sar_polarization` | SAR `VV_minus_VH`, exact evidence-schema claim | Derived polarization-related surface variation | “derived SAR evidence … polarization-related surface variation” | object/material identity, flood inference | Same as above | Same as above |
| `heterogeneity` | Optical `NDVI_local_std_5` and exact future schema claim | Derived spatial heterogeneity | Only the generic evidence statement | landscape class or cause | Referenced geometry required for location | Same as above |

`MEASURED`, `DERIVED`, `MODEL-PREDICTED`, and `INFERRED` are not interchangeable. Current public sentences are marked `INFERRED_FROM_DERIVED_EVIDENCE`; the underlying support records remain `DERIVED`. No model prediction is converted because the real evidence has no trained task-head output. Evidence strength is never called confidence.

Supported routing is deliberately small: scene presence/description, vegetation, vegetation location, water, water location, SAR/radar, heterogeneity when a valid claim exists, and supporting-evidence details. Houses, vehicles, roads, ownership, counts, exact identities/addresses, and every unrouted question return: “The current SatQuery pipeline does not provide validated evidence for that question.”

Optical and SAR claims remain separate. Their simultaneous presence permits only the statement that they provide complementary information; it does not establish a joint semantic prediction.
