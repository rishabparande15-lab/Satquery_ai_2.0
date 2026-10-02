# Phase 3AA.1 — Authorized benchmark data acquisition

Status: `PHASE3AA1_SPLIT_BLOCKED`.

VRSBench was assessed first through its author-controlled [project page](https://vrsbench.github.io/), [repository](https://github.com/lx709/VRSBench), and author-linked [Hugging Face distribution](https://huggingface.co/datasets/xiang709/VRSBench). The official repository directs users to that distribution. Its visible metadata exposes `train` and `eval`, but does not establish a development validation split distinct from a test/evaluation split. Under SatQuery's strict policy, `eval` cannot be treated as safe validation merely from its name.

The author-linked distribution lists CC-BY-4.0, while the official repository says some source imagery is derived from DOTA and restricted to academic/non-commercial use. This supports an academic-use possibility but does not independently establish the scope for the exact selected development files. Since the split gate already fails, no download was attempted.

RSVQA was considered as the allowed fallback without acquiring it. The existing authoritative RSVQA-LR audit still records unresolved release-specific terms for annotations, images, and underlying imagery. Its acquisition gate therefore remains closed.

No archive, annotation, benchmark image, test label, or test record was opened. There is no hash, image contract, linkage audit, overlap result, validation panel, inference, or metric because no dataset material was acquired. `TEST_LABEL_ACCESS`, `TEST_INFERENCE`, and `TEST_METRICS` remain zero.

The next permitted step is to obtain an authoritative VRSBench split manifest or maintainer clarification identifying a non-test validation subset and applicable file-level use terms. Then rerun this admission gate before acquisition. Alternatively, obtain a release-specific RSVQA license/use clarification and complete its independent source gate.
