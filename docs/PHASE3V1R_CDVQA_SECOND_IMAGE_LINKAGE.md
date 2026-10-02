# Phase 3V.1R — CDVQA ↔ SECOND linkage recovery

This provenance-only phase reads exactly four already-admitted annotation files:
`Train_images.json`, `Train_questions.json`, `Val_images.json`, and
`Val_questions.json`. It does not access CDVQA TEST or TEST2 annotations.

The 16:1 image-record ratio is annotation packaging. Every image record carries
`questions_ids`, every question carries `img_id`, and all sixteen records for a
pair share one `file_name`. The schema has no crop, view, T1, T2, or path
fields, so it cannot support a claim that the sixteen records represent
different physical images.

The author-maintained SECOND project page documents 512-by-512 pixel-level
annotated multi-temporal imagery and links `second_dataset.zip` and
`SECOND_train_set.rar`. It does not document archive directory structure,
T1/T2 semantics, a filename index, CDVQA stem mapping, or image-data license
terms. No archive was downloaded or inspected.

The result is `CDVQA_BLOCKED_IMAGE_LINKAGE`. The absence of a deterministic
physical path mapping also leaves temporal ordering unresolved. Image
acquisition and temporal model work remain unauthorized until the SECOND
authors or another authoritative source provides a scope-safe TRAIN/VALIDATION
mapping, documented order, and acceptable image-use terms.
