# Phase 3Q.2 controlled optical–SAR integration

Phase 3Q.2 adds an explicit, non-routed BigEarthNet S1+S2 harness at
`src.eo_vlm.optical_sar_joint.run_optical_sar_joint`. It is not imported by,
registered with, or enabled through a controller.

The harness accepts only finite canonical S1 `[2,120,120]` and S2
`[12,120,120]` tensors bearing the same expected strict BigEarthNet patch
identity and complete shared spatial metadata. Missing modalities, malformed
tensors, non-finite inputs, identity mismatches, unapproved checkpoints, and
invalid task types return structured failures. It never fills a missing
modality with zeros.

The direct path is paired input → official CROMA joint encoder → verified
Phase 3Q.1 joint projector → 16×2048 visual tokens → frozen Qwen → fixed
generation interface. Responses use the common fields: record ID, task type,
generated text, parsed answer, error status, provenance, and diagnostics.

The Phase 3Q.2 artifacts record the exact 30-record validation-panel
integration regression, S2-only regression, failure cases, and resource
measurements. The outcome is technical readiness with scientific gain still
unclear; automatic controller routing remains disabled.
