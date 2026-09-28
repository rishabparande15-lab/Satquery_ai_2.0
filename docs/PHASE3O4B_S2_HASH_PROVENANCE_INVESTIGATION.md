# Phase 3O.4B — S2 Source-Hash Provenance Investigation

Status: **PHASE3O4B_PROVENANCE_UNRESOLVED**.

The frozen Phase 3O.3 validation record reports
`c8be9921db86be788c2df86b945ba189f80fdde297cecacfb85727a38f661a50`.
The exact recorded raw-1000 source files currently reproduce
`272caa3ec89a2582b9196ac46cc4cadbafaf4e56150ccc3d1cf4afac0e273a2a`
when hashing canonical B01..B12 labels followed by each band GeoTIFF SHA-256
digest, which is the current project implementation.

The discrepancy was investigated without modifying the frozen Phase 3O.3
checkpoint, manifests, receipts, validation split, or scientific artifacts.
No model inference was executed.

The current code establishes SHA-256 and its raw-file digest concatenation
method, but Git cannot establish the exact historical generation revision or
the bytes used to produce `c8be9921`. No evidence supports calling either
value the correct historical identity. Phase 3O.4 reload validation remains
blocked until the original generation environment/receipt or an authoritative
immutable mapping establishes the historical hash input representation.
