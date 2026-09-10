# Phase 2.5 repository hygiene audit

## Audit scope

This audit covers the committed tree at Phase 2 checkpoint `4215d81`
(`4215d8117b6a43ea92583d620a641d2a186817e2`) compared with `3227014`, including
tracked paths, Git index entries, file sizes, binary extensions, generated output
directories, and credential-related text patterns. No history rewrite, reset,
force-push, or local-data deletion was performed.

## Git state before cleanup

The branch was `main`, one commit ahead of `origin/main`, with `4215d81` at
`HEAD`. The worktree contained the Phase 2 source and experiment files. The
repository object database reported approximately 131 MiB, while the tracked
working-tree files occupied approximately 134.46 MiB.

## Tracked artifact findings

`4215d81` introduced generated and downloadable artifacts alongside source:

- 233 files under `experiments/outputs/`, approximately 126.53 MiB.
- 4 satellite ZIP archives, approximately 59.84 MiB.
- 28 TIFF imagery files, approximately 59.84 MiB.
- 111 generated `.npy` feature/output arrays, approximately 10.14 MiB.
- 6 generated PNG previews, approximately 1.41 MiB.
- Repeated QA, web, hybrid, temporal, and validation JSON reports.
- Training feature-cache contents and generated split/report files.

No tracked files exceeded 50 MiB. Four tracked files exceeded 10 MiB: the
satellite ZIP archives. No tracked model checkpoint or weight file was found.

## Classification

### MUST KEEP

Source under `src/`, tests under `tests/`, experiment pipeline code, experiment
configuration, requirements, documentation, frontend assets, and small metadata
needed to understand or reproduce runs.

### SHOULD IGNORE

`experiments/outputs/`, generated pipeline output directories, downloaded TIFF
imagery and ZIP archives, `.npy`/`.npz` feature arrays, caches, model artifacts,
temporary files, and local virtual environments. These are reproducible or
machine-specific and are now protected by `.gitignore`.

### OPTIONAL

Small hand-authored experiment configuration and documentation may remain
tracked. Generated reports may be retained outside Git as release evidence or
published separately when a particular report is intentionally selected.

### MUST REMOVE FROM GIT

No committed credentials or private keys were found. No file required category-D
removal.

## Secret and credential audit

Tracked text files were scanned for API keys, tokens, passwords, service-account
material, private keys, OAuth secrets, Earth Engine credentials, Hugging Face
tokens, and provider credential variables. Matches were documentation, source
identifiers, or deliberate redaction-test strings; no actual secret value was
found. `.env.example` contains local paths only and remains safe. Credentials
were not rotated or revoked.

## `.gitignore` assessment

The original ignore file covered only `.env`, Python caches, and bytecode. It now
also protects environment variants while allowing `.env.example`, generated
experiment outputs, generated pipeline outputs, downloaded imagery/archives,
feature arrays, model artifacts, caches, virtual environments, logs, and temp
files. The whole `experiments/` directory is not ignored; source, configuration,
and documentation remain visible to Git.

## Cleanup actions

The ignore rules were updated and inappropriate generated/downloaded artifacts
were removed from Git tracking with a normal follow-up cleanup change. Local
files were preserved on disk. The cleanup removes tracked output contents under
`experiments/outputs/` and tracked `outputs/` directories under experiment
pipelines, including their generated imagery, archives, arrays, reports, and
previews. No source code, tests, configuration, or documentation was removed.

## Validation

After cleanup, run the full Python suite, frontend tests, Python compilation,
and `git diff --check`. Phase 1 and Phase 2 tests are included in the full suite.
The local dataset, CROMA checkpoint, and generated outputs remain available for
the real-data checks because they were not deleted from disk.

## Remaining risks

Git history still contains the removed artifacts in checkpoint `4215d81`; this
audit intentionally does not rewrite history. The repository also relies on
external dataset and checkpoint paths documented in `.env.example`. Future large
research evidence should be stored in release storage or Git LFS only after an
explicit reproducibility decision.

## Phase boundary recommendation

Phase 2.5 is complete after the cleanup commit and validation. Do not begin
Phase 3 until the cleanup commit is reviewed and the external-data provisioning
plan is agreed. This audit adds no retrieval, modeling, temporal, UI, or agent
functionality.