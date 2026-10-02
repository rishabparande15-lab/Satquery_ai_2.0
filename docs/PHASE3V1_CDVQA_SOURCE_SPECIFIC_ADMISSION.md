# Phase 3V.1 — CDVQA source-specific admission

Phase 3V.1 inspected only the official CDVQA `Train_*` and `Val_*` JSON
annotations at revision `cc5893123dd32326de38745b65d2ffe45055937b`. It did
not open, download, parse, or derive manifests for TEST or TEST2 payloads.

The annotation schema explicitly joins questions to image records with
`img_id`, question records to answers with `answers_ids`, and answers back to
questions with `question_id`. The `type` field supplies all eight published
question families. A single `file_name` is present per annotation image record;
neither separate T1/T2 paths nor a temporal-order field is present.

The official SECOND project page is an author-maintained source and points to
Google Drive. It documents 512-by-512, pixel-level annotated bi-temporal
imagery, but does not publish a CDVQA TRAIN/VALIDATION-only linkage,
temporal-order mapping, or scoped use terms on that page. Acquiring the full
source would risk importing identities that are reserved for CDVQA TEST, so no
imagery was downloaded.

Therefore the correct admission result is
`CDVQA_ANNOTATIONS_ADMITTED_IMAGES_PENDING`, not full temporal-development
admission. No temporal model, training, API, frontend, production route, or
Playwright run is authorized. The next step is to obtain authoritative
TRAIN/VALIDATION-only SECOND linkage and temporal-order documentation before
any imagery acquisition.
