# Scientific limitations

- No SOTA, benchmark-superiority, or sensor-generalization claim is made.
- Single-image VQA results are internal validation only.
- Optical/SAR integration is technically valid but did not outperform the S2-only control.
- Change description is functional text generation only; it has no change mask, area estimate, or benchmark metric receipt.
- Scene descriptions use an explicit RGB image/rendering; they do not infer sensor identity, have no calibrated confidence, and have no benchmark evaluation receipt for this SatQuery integration.
- Grounding remains unavailable: SatQuery deliberately returns a structured blocked result rather than fabricate a box, mask, or confidence.
- Cartosat-2S and RISAT require explicit adapter validation. Mismatched inputs must receive `SENSOR_DOMAIN_NOT_VALIDATED`.
- Generic raster inspection does not establish Cartosat-2S or RISAT model generalization, band equivalence, or benchmark performance.
