"""Real unified API smoke for Phase 3X scene description."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
payload = {
    "requested_task": "SINGLE_IMAGE_SCENE_DESCRIPTION",
    "query": "Describe this image.",
    "patch_id": "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11",
    "inputs": [{"role": "SINGLE", "modality": "rgb", "input_id": "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11", "source": "approved_local_patch"}],
}
request = Request(os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8770") + "/api/v1/query", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=900) as response:
    result = json.loads(response.read().decode())
out = ROOT / "artifacts" / "final" / "sih" / "phase3x" / "phase3x_scene_api_result.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"http_status": 200, "request": payload, "response": result}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"http_status": 200, "route": result.get("route"), "status": result.get("status"), "answer": result.get("answer")}, indent=2))
