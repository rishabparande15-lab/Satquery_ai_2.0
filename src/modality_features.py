"""Single-modality statistics plus an explicitly versioned legacy joint schema."""
import numpy as np
from .gee_features import LocalRasterFeatureProvider, _stats, _index
from .dataset_loader import OPTICAL_BANDS, SAR_BANDS


def extract(arrays):
    if set(arrays) == {"optical", "sar"}:
        vector, report = LocalRasterFeatureProvider().extract(arrays["optical"], arrays["sar"], list(OPTICAL_BANDS), list(SAR_BANDS))
        report["schema"] = "legacy_joint_62"
        report["limitations"] = ["Legacy SAR relationship features are clipped to [-1,1] and are not physical dB ratios. Retained for checkpoint/cache compatibility."]
        return vector, report
    modality = next(iter(arrays))
    array = arrays[modality]
    bands = OPTICAL_BANDS if modality == "optical" else SAR_BANDS
    values, counts, formulas = {}, {}, {}
    for band, channel in zip(bands, array):
        stats, count = _stats(channel, f"{modality}_{band}")
        values.update(stats)
        counts[f"{modality}_{band}"] = count
    if modality == "optical":
        by_band = dict(zip(bands, array))
        for name, a, b in (("NDVI", "B08", "B04"), ("NDWI", "B03", "B08"), ("MNDWI", "B03", "B11"), ("NDBI", "B11", "B08")):
            key = f"index_{name}_mean"
            values[key], counts[key] = _index(by_band[a] - by_band[b], by_band[a] + by_band[b], name)
            formulas[name] = f"({a}-{b})/({a}+{b})"
    else:
        values["sar_VV_minus_VH_mean"] = float(np.mean(array[0].astype("float64") - array[1]))
        counts["sar_VV_minus_VH_mean"] = int(array[0].size)
        formulas["sar_VV_minus_VH_mean"] = "mean(VV - VH), in input units; no dB division"
    vector = np.array(list(values.values()), dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Physical feature calculation overflowed.")
    return vector, {"dimension": int(vector.size), "feature_names": list(values), "feature_values": values,
                    "valid_pixel_counts": counts, "formulas": formulas, "schema": f"{modality}_v1", "limitations": []}
