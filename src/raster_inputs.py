"""Assemble only requested modalities using the established reprojection path."""
from datetime import datetime
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from .config import get_settings
from .dataset_loader import discover_samples, OPTICAL_BANDS, SAR_BANDS, _read_to_grid
from .input_validation import inspect_raster, check_aoi_intersection


def assemble(plan, retrieval, request):
    arrays, metadata, references = {}, {}, []
    local = retrieval["source"] == "local development provider"
    if local:
        sample = next(s for s in discover_samples(get_settings().dataset_root) if s.patch_id == request["sample_id"])
        paths = {"optical": sample.optical_paths, "sar": sample.sar_paths}
        reference = sample.optical_paths.get("B02") if "optical" in plan["modalities"] else sample.sar_paths.get("VV")
        if reference is None:
            raise ValueError("Missing reference band for requested modality.")
        with rasterio.open(reference) as grid:
            for modality in plan["modalities"]:
                bands = OPTICAL_BANDS if modality == "optical" else SAR_BANDS
                if set(paths[modality]) != set(bands):
                    raise ValueError(f"Missing or unexpected {modality} bands.")
                for band in bands:
                    info = inspect_raster(paths[modality][band])
                    if not info["crs"] or not info["finite"]:
                        raise ValueError(f"{modality} band {band} has missing CRS or invalid pixels.")
                arrays[modality] = np.stack([_read_to_grid(paths[modality][band], grid, Resampling.bilinear) for band in bands])
                info = inspect_raster(reference)
                info.update(band_count=len(bands), band_order=list(bands), shape=list(arrays[modality].shape[1:]))
                match = re.search(r'20\d{6}T\d{6}', next(iter(paths[modality].values())).name)
                info["acquisition_date"] = datetime.strptime(match.group(0)[:8], "%Y%m%d").date().isoformat() if match else None
                info["sar_units"] = "dB" if modality == "sar" else None
                metadata[modality] = info
                references.append({"kind": modality, "reference": f"local:{sample.patch_id}:{modality}", "status": "local data", "bands": list(bands)})
    else:
        for modality in plan["modalities"]:
            path = retrieval["assets"][modality]
            info = inspect_raster(path)
            bands = list(OPTICAL_BANDS if modality == "optical" else SAR_BANDS)
            with rasterio.open(path) as ds:
                described = list(ds.descriptions)
                if all(described) and set(described) == set(bands):
                    order = [described.index(band) + 1 for band in bands]
                elif request.get("band_order_confirmed") is True and not any(described):
                    order = list(range(1, len(bands) + 1))
                else:
                    raise ValueError(f"{modality} requires canonical band descriptions or explicit band-order confirmation.")
                arrays[modality] = ds.read(order, out_dtype="float32", masked=True).filled(np.nan)
            info["band_order"] = bands
            metadata[modality] = info
            references.append({"kind": modality, "reference": f"upload:{modality}", "status": "user-provided data"})
    for modality, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{modality} has non-finite pixels after alignment.")
        check_aoi_intersection(plan.get("aoi"), metadata[modality])
        acquired = metadata[modality].get("acquisition_date")
        if plan.get("start_date") or plan.get("end_date"):
            if not acquired:
                raise ValueError(f"{modality} acquisition date is unknown; requested date filter cannot be verified.")
            if (plan.get("start_date") and acquired < plan["start_date"]) or (plan.get("end_date") and acquired > plan["end_date"]):
                raise ValueError(f"{modality} acquisition date falls outside the requested date range.")
    return arrays, metadata, references
