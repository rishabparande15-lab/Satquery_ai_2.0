# Phase 3V.1A — authoritative SECOND resolution

The source-resolution audit found no primary CDVQA or SECOND loader that maps
an admitted CDVQA `file_name` to physical image paths. The official CDVQA
repository is annotation-only. The author-maintained SECOND page links data
archives but does not publish archive trees, T1/T2 directory semantics,
stem-mapping rules, or image-data use terms.

Two official CDVQA issues were inspected. Neither has a maintainer response
that clarifies linkage, ordering, download structure, or licensing.

The official VisTA implementation contains a supporting, non-primary loader
convention: `im1/<file_name>` and `im2/<file_name>`. It does not state which
member is earlier or later, and cannot establish that those paths are the
authoritative CDVQA-to-SECOND mapping. This evidence therefore does not clear
the admission gate.

The result is `CDVQA_AWAITING_AUTHOR_CONFIRMATION`. No imagery was downloaded,
no TEST/TEST2 material was accessed, and temporal model work remains blocked.
The prepared contact package requests the exact missing facts needed for a
scope-safe TRAIN/VALIDATION-only acquisition.
