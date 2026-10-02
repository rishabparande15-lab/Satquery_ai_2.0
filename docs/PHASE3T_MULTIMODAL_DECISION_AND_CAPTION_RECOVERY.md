# Phase 3T multimodal decision and caption diagnosis

Phase 3T reuses the exact fixed 30-record non-TEST panel and its deterministic
shuffle map. It is a read-only diagnosis: no checkpoint, prompt, dataset,
threshold, or historical artifact is changed.

Stored Phase 3Q.3 traces show that shuffled and zeroed S1/S2 conditions move
candidate-level margins, but produce zero generated-output changes. Binary and
MCQ generation can diverge from full candidate-sequence ranking. The final
Phase 3R caption interface produced zero nonempty outputs; historical traces
do not contain token-level EOS ranks, so no EOS value is invented.

The preregistered Phase 3T gate selects `NO_TRAINING`. The evidence does not
justify assuming that one projector-only objective can overcome frozen-Qwen
language/decoding priors, weak decision-level visual dependence, and caption
collapse. The resulting classification is `PROJECTOR_ONLY_INSUFFICIENT`.
SatQuery v1 remains frozen; any Qwen LoRA investigation requires separately
authorized future research.
