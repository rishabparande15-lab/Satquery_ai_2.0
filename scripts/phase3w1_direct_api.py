"""Persistent direct unified API verification for the two approved local inputs."""
from __future__ import annotations
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final" / "agent" / "phase3w1r_root_recovery"
BASE = os.environ.get("SATQUERY_BASE_URL", "http://127.0.0.1:8768")

def call(payload):
    request = Request(BASE + "/api/v1/query", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=900) as response:
        return {"http_status": response.status, "body": json.loads(response.read())}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    single_id = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
    pair_id = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
    single = call({"query": "Is water visible in this image?", "patch_id": single_id, "task_type": "binary_qa", "inputs": [{"role": "SINGLE", "modality": "s2", "input_id": single_id, "source": "approved_local_train"}]})
    paired = call({"query": "Use the SAR and optical information together. Is water visible?", "patch_id": pair_id, "task_type": "binary_qa", "inputs": [{"role": "S1", "modality": "s1", "input_id": "S1B_IW_GRDH_1SDV_20170717T064605_29UPV_35_22", "source": "approved_local_validation"}, {"role": "S2", "modality": "s2", "input_id": pair_id, "source": "approved_local_validation"}]})
    (OUT / "phase3w1r_api_results.json").write_text(json.dumps({"single_image": single, "optical_sar": paired}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
