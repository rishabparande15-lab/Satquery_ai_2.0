# SatQuery final architecture

`User file → generic raster inspector → explicit sensor/role declaration → sensor adapter compatibility gate → Unified Web UI / POST /api/v1/query → SatQueryAgent → input configuration classifier → query intent classifier → capability registry → route-specific validation → specialist → normalized response → provenance, warnings, confidence policy, execution trace → JSON export`

Specialists: S2 projector + frozen Qwen (single VQA); CROMA joint + projector + Qwen (optical/SAR); Chg2Cap (bi-temporal change description); AdaptLLM remote-sensing Qwen2-VL-2B (single-image scene description). Grounding remains blocked.

The generic inspector does not infer sensor identity, reorder bands, resample rasters, or load a specialist. Cartosat-2S and RISAT adapters are metadata/provenance foundations only and return `SENSOR_DOMAIN_NOT_VALIDATED` until sensor-specific model validation exists.
