# Evidence claims and strength rules

Claims are created by deterministic feature rules, not an LLM. A candidate token must be at or above the scene's feature p75 and above the feature-specific physical floor: NDVI > 0.2, NDWI > 0, NDBI > 0, and VV−VH above its p75. Candidates are grouped with four-connectivity.

Strength depends on valid coverage, total supporting-token fraction, and largest connected-region fraction:

- `UNAVAILABLE`: coverage < 0.5 or no support.
- `STRONG`: support ≥ 0.35 and largest region ≥ 0.20 of all tokens.
- `MODERATE`: support ≥ 0.15 and largest region ≥ 0.08.
- `WEAK`: some traceable support below those thresholds.

These are transparent descriptive rules, not probabilities or statistical confidence. Sensor views stay separate. A joint claim requires an explicit cross-sensor semantic agreement rule; joint CROMA presence or concatenation alone is insufficient.

Sample `61_39` produced three actual, moderate claims: vegetation-related spectral evidence (57 tokens, five regions), water-related spectral evidence (53 tokens, two regions), and elevated SAR polarization-contrast evidence (57 tokens, ten regions). No built-up claim passed its positive NDBI floor, so none was emitted. The statements deliberately say “evidence,” not detected objects or land-cover labels.

Every claim carries feature ID, sensor, evidence type, token indices, exact values, region IDs, source artifact and uncalibrated status. `validate_evidence` rejects empty support, missing source artifacts, support/value length mismatches, missing region references, invalid strength values, or non-row-major token objects.
