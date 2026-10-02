# Phase 3AC — Runtime and Deployment Hardening

Start SatQuery through `scripts/start_satquery.ps1`. It loads the existing `.env`, runs a non-loading preflight, stops with explicit asset errors, then starts the loopback backend. `scripts/preflight_satquery.ps1` can be used independently.

Configuration precedence is environment, then `.env`, then project-safe defaults. Required BigEarthNet, LEVIR validation, CROMA, projector, and Chg2Cap assets are checked without exposing their paths through browser APIs. `GET /api/v1/health` is lightweight. `GET /api/v1/ready` summarizes readiness and route status.

Models remain lazy-loaded and cached only within their existing route controllers. No model fallback is allowed. Upload staging remains bounded and temporary; runtime logs rotate at 2 MiB with three backups. The service binds to loopback only and supports clean Ctrl+C shutdown.

This release is offline-capable when all configured local assets are already present. It does not download assets at startup. The four approved routes retain their prior real-browser smoke results. A new repeated heavyweight-inference memory smoke was deliberately not rerun in this hardening pass, so Phase 3AC is correctly classified as runtime-partial rather than claiming an unmeasured memory result.
