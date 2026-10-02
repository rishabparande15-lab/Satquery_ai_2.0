# SIH submission checklist

## Documentation and presentation

- [x] README describes the frozen release candidate and scientific limits.
- [x] Presentation content covers the verified routes, architecture, results, and limitations.
- [x] Final report outline is ready for the written submission.
- [x] Architecture diagram content is ready and contains no obsolete routes.
- [x] Presenter demo script is ready.
- [x] Viva and judge question bank contains 30+ questions.
- [x] SIH compliance matrix and scientific limitations are linked from the project index.

## Demonstration readiness

- [x] Approved non-test demo inputs are identified.
- [x] Four primary real browser flows passed.
- [x] Grounding safety flow returns `BLOCKED` without fabricated geometry.
- [x] Screenshots and JSON export receipts are retained locally.
- [x] Browser primary-flow error counts are zero.
- [x] Demo backup consists of the Phase 3X.1 screenshots and receipt artifacts.

## Verification

- [x] Focused Python suite: 37 passed.
- [x] Frontend suite: 7 passed.
- [x] Browser rehearsal: four primary flows plus one blocked safety flow passed.
- [x] Requirements files are present.
- [x] Test access counters remain zero for the protected datasets and splits.

## Repository hygiene

- [x] `.gitignore` excludes local datasets, model checkpoints, Hugging Face caches, runtime logs, temporary files, `.env`, and common credentials.
- [x] Local receipts are retained but `artifacts/final/` remains ignored.
- [x] No dataset or checkpoint should be committed.
- [x] Secret scan completed without confirmed secrets in tracked source/configuration files.
- [ ] Review existing uncommitted work before a release commit.
- [ ] Commit, push, and tag only after explicit user authorization.

## Manual SIH tasks

- [ ] Transfer the presentation content into the required SIH PPT template, if one is supplied.
- [ ] Draft the final report from the report outline and add formal citations.
- [ ] Verify the target presentation machine has the approved local assets, GPU runtime, and browser.
- [ ] Rehearse the final demo using only approved non-test inputs.
- [ ] Record a demonstration video backup if the submission process requires one.
