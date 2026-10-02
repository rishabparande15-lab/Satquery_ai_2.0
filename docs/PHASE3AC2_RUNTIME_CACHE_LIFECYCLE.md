# Phase 3AC.2 — Runtime Cache and Model Lifecycle

`SpecialistRuntime` centralizes specialist construction and GPU residency. It serializes heavy inference, keeps a controller for repeated same-route requests, and evicts the prior heavy family before switching routes. Release removes references, runs garbage collection, and clears unreferenced CUDA cache only after the active request completes.

The optical-SAR reload was caused by `SatQueryV1Controller()` being constructed inside `optical_sar_executor`. Three post-remediation optical-SAR requests constructed its controller once and returned the unchanged answer `Yes` each time. A VQA → scene → optical-SAR → temporal → VQA sequence completed in one healthy server session with a maximum observed allocation of 8,347,854,848 bytes and reservation of 8,489,271,296 bytes.

No model, checkpoint, prompt, preprocessing, or generation setting changed. Checkpoint digests remained unchanged and output receipts matched for the approved deterministic inputs. A new scene browser check passed; the remaining procedural item is a fresh all-four-route browser replay after the lifecycle change.
