# SatQuery sensor compatibility matrix

| Input | Route | State | Evidence / boundary |
| --- | --- | --- | --- |
| Sentinel-2 | Single-image VQA | Validated development path | Explicit 12-band canonical S2 contract only. |
| Sentinel-1 + Sentinel-2 | Optical-SAR analysis | Validated development path | Approved paired development route only. |
| Explicit RGB | Scene description | Compatible with limitations | Explicit RGB input; sensor identity is not inferred. |
| LEVIR-CC A/PRE + B/POST | Temporal change description | Validated development path | Chg2Cap development-only contract. |
| Cartosat-2S | Raster inspection / provenance | Validated | User declaration and metadata inspection only. |
| Cartosat-2S | Model inference | Domain validation pending | `SENSOR_DOMAIN_NOT_VALIDATED`; it is not treated as Sentinel-2. |
| RISAT | Raster inspection / provenance | Validated | User declaration and metadata inspection only. |
| RISAT | CROMA or other model inference | Domain validation pending | `SENSOR_DOMAIN_NOT_VALIDATED`; it is not treated as Sentinel-1. |

File readability never establishes model compatibility. Equal modality does not establish equal sensor domain. External pair inspection also retains `COREGISTRATION_NOT_VERIFIED` until authoritative spatial evidence is admitted.
