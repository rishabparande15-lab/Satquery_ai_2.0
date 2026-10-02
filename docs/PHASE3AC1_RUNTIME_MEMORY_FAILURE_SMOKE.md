# Phase 3AC.1 — Bounded Runtime Memory and Failure Smoke

Three sequential approved non-test requests were run in isolated server processes for each route. VQA, scene description, and temporal change description loaded once then showed identical post-request CUDA allocation/reservation values on requests two and three. Optical-SAR completed all three requests but reloaded its heavy controller each time; its memory result is therefore inconclusive rather than a claim of stability.

An additional mixed-route process showed GPU pressure when VQA and scene specialists remained cached and optical-SAR subsequently loaded. The optical-SAR request returned a structured HTTP 422 while the health endpoint remained available. No forced OOM, data change, checkpoint change, or model change was performed.

Bad-dataset preflight correctly returned `BLOCKED` in an isolated environment and returned `READY` after restoring normal configuration. TEST access remained zero. The correct result is `PHASE3AC1_MEMORY_INCONCLUSIVE`; runtime architecture changes are not made during a verification-only phase.
