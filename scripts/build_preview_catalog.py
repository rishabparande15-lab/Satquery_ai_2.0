"""Print a frontend-only catalog of real local sample display previews.

Does not alter backend modules, source imagery, or model inputs.
"""
import base64
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.config import get_settings
from src.dataset_loader import discover_samples, load_sample

catalog = {}
for sample in discover_samples(get_settings().dataset_root):
    item = load_sample(sample)
    views = {}
    for role, data in (("optical", np.moveaxis(item.raw_optical[[3,2,1]], 0, -1)), ("sar", item.raw_sar[0])):
        lo, hi = np.percentile(data, [2,98])
        display = np.uint8(np.clip((data-lo) / max(hi-lo, 1e-6), 0, 1) * 255)
        stream = io.BytesIO()
        Image.fromarray(display).save(stream, format="PNG")
        views[role] = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    catalog[sample.patch_id] = {"views": views, "crs": item.metadata["crs"], "resolution": item.metadata["resolution"], "shape": item.metadata["shape"],
                              "source": "Local development sample", "rendering": "RGB B04/B03/B02; SAR VV grayscale; 2-98% display stretch. Not a prediction."}
print(json.dumps(catalog))
