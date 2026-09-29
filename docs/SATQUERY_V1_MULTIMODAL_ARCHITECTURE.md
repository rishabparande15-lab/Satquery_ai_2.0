# SatQuery v1 multimodal architecture

The primary SatQuery v1 route requires a paired BigEarthNet Sentinel-1
`[2,120,120]` VV/VH tensor and Sentinel-2 `[12,120,120]` tensor. It validates
identity, spatial metadata, shape, and finite values before running official
paired CROMA, the verified Phase 3Q.1 joint projector, frozen Qwen, generation,
parsing, and structured provenance.

The entrypoint is `SatQueryV1Controller.run_satquery`; its route is
`MULTIMODAL_S1_S2`. Missing S1 or S2 fails closed. S2-only remains an explicit
baseline, never an automatic fallback.
