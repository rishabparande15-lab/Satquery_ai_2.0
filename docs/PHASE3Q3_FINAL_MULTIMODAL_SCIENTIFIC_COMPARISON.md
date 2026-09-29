# Phase 3Q.3 final multimodal scientific comparison

Phase 3Q.3 is the closing SatQuery v1 S1+S2 comparison. It reuses the exact
Phase 3Q.1 validation panel: 30 BigEarthNet records, comprising ten Binary,
ten MCQ, and ten Caption prompts. It uses the saved deterministic shuffle map
and accesses no TEST records.

The inference-only conditions are S2-only correct; correct S1+S2; shuffled
S1; shuffled S2; fully shuffled; zero S1; and zero S2. Qwen, CROMA, the
historical S1/S2 projectors, and the Phase 3Q.1 joint projector are frozen and
fingerprint-checked after the run.

The route gate is preregistered in `phase3q3_preregistered_gate.json`. It
uses observed fixed-panel outputs and task correctness only; it does not
claim benchmark performance or subjective caption quality. Candidate-score
and target-margin values are retained in the per-record artifact as diagnostic
evidence, but are not substituted for generated-answer behavior in the route
decision.

The recorded outcome is `OPTICAL_SAR_TECHNICALLY_VALID_S2_DOMINANT` with
`KEEP_S2_DEFAULT_AND_RETAIN_FUSION_FOR_RESEARCH_ONLY`. The joint path remains
an auditable research artifact; it is not connected to automatic routing.
