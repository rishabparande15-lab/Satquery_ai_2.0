# Phase 3AA — RSVQA and VRSBench benchmark evaluation admission

Status: `PHASE3AA_BLOCKED_DATA`.

This phase performed an authoritative-source and local-data admission audit only. It did not download a benchmark, parse a benchmark annotation, inspect a benchmark image, run inference, train, alter a checkpoint, tune a prompt, or calculate a score. `TEST_IMAGE_ACCESS`, `TEST_LABEL_ACCESS`, `TEST_INFERENCE`, and `TEST_METRICS` are all zero.

## Source facts

The [RSVQA project](https://rsvqa.sylvainlobry.com/) is the author-controlled source for RSVQA-LR, RSVQA-HR, and RSVQAxBEN. The project describes image/question/answer datasets and links separate code/data routes. The existing Phase 3N.4 audit remains controlling for RSVQA-LR: author-linked Zenodo release metadata was identified, but release-specific image/annotation/underlying-imagery terms, local linkage, split membership, and official evaluator admission remain unresolved. Its variants must not be treated as interchangeable.

[VRSBench](https://vrsbench.github.io/) and its [official repository](https://github.com/lx709/VRSBench) describe image captioning, visual grounding, and open-ended VQA. The official repository identifies CC-BY-4.0 for text annotations and notes that some DOTA-source imagery is academic-use only/non-commercial. It documents evaluation helpers, including automatic/GPT-based caption and GPT-based VQA evaluation. This is not yet a local evaluation contract for SatQuery.

## Local audit and compatibility

No RSVQA or VRSBench dataset directory, manifest, image archive, annotation file, or prior authorized benchmark receipt was found in the workspace or configured project-data roots. There are therefore no admitted train/validation records, no metrics, and no benchmark evidence.

RSVQA input compatibility is unknown until the exact selected variant's image modality, preprocessing, answer schema, split manifest, use terms, and evaluator are admitted. VRSBench VQA and captioning are candidates for separately validated adapters, but neither is directly enabled by a generic RGB upload. Grounding remains blocked because SatQuery has no validated grounding specialist or coordinate contract.

`src.benchmark_admission` now provides a lightweight record receipt contract and rejects `test` split records before evaluation. It does not read data or invoke a model.

## Next gate

Obtain an authoritative, use-permitted RSVQA or VRSBench development release with immutable revision, train/validation manifest, image↔annotation linkage, and evaluation protocol. Then rerun admission and execute one deterministic validation panel without tuning. Benchmark test content remains excluded unless a later explicit final-evaluation authorization changes that policy.
