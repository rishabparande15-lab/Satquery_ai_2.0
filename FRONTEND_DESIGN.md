# SatQuery frontend redesign

## Inspection and safe implementation plan

Current architecture: src/static/index.html, style.css and app.js are served by the existing Python API's fixed static-route allowlist. No framework or frontend build step. The form uses multipart uploads followed by a JSON analysis request. Existing Node state tests cover error, timeout, reset and duplicate-submit recovery; pytest covers the backend contracts.

Design history: the only Git commit contains README.md. The repository and its parent contain no older HTML/CSS/React/Figma design or UI screenshot. No previous design can be honestly recovered. Existing optical/SAR reference PNGs and local rasters are available. The current design is a green, vertically stacked form/results page with no app navigation, large viewer or history. Retain its useful green status accent, rounded panels and clear scientific disclosures; replace its dense form-first hierarchy with an image-led, graphite research workspace.

Reuse: existing request(), schema handling, upload roles, timeouts, busy guard, reset, evidence/trace/report rendering contract; sample IDs and actual imagery. Redesign: shell, navigation, hierarchy, typography, mode selection, uploads, large viewer, validation metadata, results, comparison, history and status presentation.

Immutable API contracts:

- GET /api/health -> status/service/busy; GET /api/samples -> samples/source.
- POST /api/upload uses multipart optical/sar/before/after roles; returns opaque file IDs.
- POST /api/analyze retains query, analysis_type, aoi, start_date, end_date, files, sample_id and band_order_confirmed fields. Never send a new unsupported mode enum.
- Analysis reports contain status, analysis_id, validation, features, data_cube, evidence, confidence, execution_trace, runtime_seconds and device. HTTP 422 may include a downloadable rejected report.
- GET /api/report/{analysis_id} remains the persistent server download.
- Backend files, CSP and routes remain unchanged. Assets must be embedded in the three allowed frontend files or rendered locally.

Implementation sequence: establish tokens and app shell; redesign home and workspace; add mode cards and accessible drop zones; use actual local image previews with provenance; add validation/results/evidence/trace panels; implement local-browser report history with no invented server listing; provide side-by-side/slider input viewing only when previews exist, clearly separating visual comparison from algorithmic change detection; verify responsive behavior and preserve all old tests; run live backend workflows.

Constraints: classification head and temporal algorithm remain unavailable. The feature-extraction UI choice maps to the existing automatic task parser. A browser history entry stores only ID/query/time/status locally, retrieving the full report through the existing download route. No fabricated maps, percentages, confidence, current-stage telemetry or prior analyses.
