# Pass 5D spatial/evidence comparison

Version A is a scene predictor over 62 physical statistics and pooled joint CROMA. Although upstream CROMA has spatial tokens, Version A's prediction input contains no explicit token-local physical value or deterministic physical region. It can answer “what is predicted for the scene,” but cannot trace vegetation-, water-, texture-, or SAR-related statements to physical token values.

Version B retains the same predictor inputs and adds selected pixel/local evidence aligned to 225 tokens. The `61_39` evidence artifact provides 12 selected token features, exact pixel/projected extents, valid counts, sensor identity, 17 deterministic regions and three supported claims. Actual claims are vegetation-related spectral evidence, water-related spectral evidence, and SAR polarization-contrast evidence. No built-up claim passed its rule. These are evidence regions, not segmentation.

| Deterministic question proxy | Version A | Version B |
|---|---|---|
| What is present? | Scene prediction only | Same scene prediction plus qualified physical evidence |
| Where is vegetation evidence? | Unavailable from pooled physical input | NDVI tokens and regions |
| Where is water evidence? | Unavailable from pooled physical input | NDWI tokens and regions |
| Homogeneous or heterogeneous? | Scene statistics only | Token distributions and local standard deviation |
| What does SAR indicate? | Scene statistics/latent pooled representation | VV, VH, VV−VH and local texture by token |

Version B therefore wins the spatial evidence comparison through traceable information, not merely more dimensions. No token-level prediction, segmentation accuracy, calibrated confidence, or object detection is claimed. Proxy responses are stored in `experiments/pass5d/comparison/question_proxy.json`; they are deterministic templates, not LLM output.
