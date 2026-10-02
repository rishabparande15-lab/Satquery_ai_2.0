# Phase 3V.2B — Real browser verification

Status: `PHASE3V2B_COMPLETE`

The production-style verification chain was exercised with a real local SatQuery application and a real validation-only LEVIR-CC pair:

`Google Chrome -> BI-TEMPORAL frontend -> POST /api/v1/temporal -> Chg2Cap -> rendered change description`

## Runtime

- Browser: existing Google Chrome 154.0.8037.58, headless.
- Automation: installed Python Playwright. No browser runtime was downloaded.
- Application URL during verification: `http://127.0.0.1:8766`.
- Pair: `val_000001.png`; T1 was `A / PRE` and T2 was `B / POST`.

## Observed result

The real request used the question “What changed between these two images?” and received HTTP 200 from `/api/v1/temporal`. The route was `TEMPORAL_CHANGE_DESCRIPTION`, the selected specialist was `Chg2Cap`, and the model-generated output was: “a row of houses is built at the top of the scene”.

The UI showed the before and after previews, generated description, selected route, model provenance, and PRE_POST order. The happy path had zero console errors, zero page errors, and zero failed requests. The separate negative tests intentionally generated HTTP 422 and 415 responses; their browser resource messages are retained as expected negative-path evidence and do not affect the happy-path result.

## Safety and scope

Only the admitted validation pair was used. `TEST_CONTENT_ACCESS`, `TEST_LABEL_ACCESS`, `TEST_INFERENCE`, and `TEST_METRICS` all remain zero. The checkpoint and vocabulary were not changed.

## Limitations

Chg2Cap generates a language description only. It does not produce a change mask, bounding boxes, or area measurements.
