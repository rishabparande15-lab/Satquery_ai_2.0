# SatQuery final verified results

These results describe separate, verified checks. They are not an aggregate benchmark score.

| Area | Evidence | Result | Interpretation |
| --- | --- | --- | --- |
| Single-image binary VQA | Internal 30-record panel | 17/30, 56.7% | Internal validation only |
| Single-image MCQ | Internal 30-record panel | 8/30, 26.7% | Internal validation only |
| Optical + SAR | Real browser route | Technically valid; no gain over S2-only demonstrated | Do not claim fusion superiority |
| Temporal change description | Development smoke panel | 5/5 nonempty generated captions | Functional check, not benchmark evaluation |
| Scene description | Real browser route | Nonempty generated description | Functional check, no benchmark receipt |
| Agent routing | Final browser rehearsal | 4/4 primary routes correct | Input and query contracts selected the expected specialist |
| Grounding | Real browser safety test | Blocked safely | No fabricated box, mask, confidence, or fallback |
| Browser quality | Four primary browser flows | 0 console errors, 0 page errors, 0 failed network requests | Verified local happy paths |
| Focused regression | Python and frontend suites | 37 Python passed; 7 frontend passed | Focused release regression suite |

The Phase 3X.1 receipt records the actual live outputs and route provenance: [PHASE3X1_FINAL_DEMO_REHEARSAL.md](PHASE3X1_FINAL_DEMO_REHEARSAL.md).
