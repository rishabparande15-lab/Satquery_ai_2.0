# SatQuery benchmark compatibility matrix

| Benchmark / task | Authoritative source | Local development data | Existing route status | Evaluation state |
| --- | --- | --- | --- | --- |
| RSVQA VQA | [RSVQA project](https://rsvqa.sylvainlobry.com/) | No | Requires source-specific image/preprocessing, answer-format, and evaluator admission | Blocked: data unavailable |
| VRSBench VQA | [VRSBench project](https://vrsbench.github.io/) / [repository](https://github.com/lx709/VRSBench) | No | Requires source-specific route and open-answer evaluator admission | Blocked: data unavailable |
| VRSBench captioning | Same | No | Candidate: `SINGLE_IMAGE_SCENE_DESCRIPTION`; official references/evaluator required | Blocked: data unavailable |
| VRSBench grounding | Same | No | `SINGLE_IMAGE_GROUNDING` remains blocked | Blocked: no validated grounding specialist |

Application smoke results and internal validation are not public benchmark results. `TEST_ACCESS = 0` is maintained for every row.
