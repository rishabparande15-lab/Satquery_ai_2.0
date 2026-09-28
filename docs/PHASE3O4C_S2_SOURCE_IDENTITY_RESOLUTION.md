# Phase 3O.4C — S2 Source Identity Resolution

Status: **PHASE3O4C_EXPLAINED**. The historical `c8be9921…` hash was reproduced exactly, without changing frozen artifacts.

`scripts/train_phase3g_projector.py::_area_source_hash` defines the historical representation: SHA-256 over UTF-8 compact JSON of ordered `(band, file_sha256_hex)` pairs for B01 through B12. For the exact frozen validation asset the input was 901 bytes and reproduces `c8be9921db86be788c2df86b945ba189f80fdde297cecacfb85727a38f661a50`.

The prior `272caa…` value is a different, non-historical representation: labels concatenated with binary digest bytes. It is not a source-identity conflict.

No Phase 3O.4 model validation was executed. No model inference or training occurred. No benchmark data or test data was accessed. The frozen Phase 3O.3 artifacts were not rewritten.
