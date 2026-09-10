import numpy as np
import rasterio
from rasterio.enums import Resampling


def load_croma_inputs(paths: dict[str, dict], output_size: int = 120) -> tuple[np.ndarray, np.ndarray]:
    optical_bands = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
    sar_bands = ("VV", "VH")

    def read_group(group: dict, bands: tuple[str, ...]) -> np.ndarray:
        arrays = []
        for band in bands:
            with rasterio.open(group[band]) as dataset:
                arrays.append(dataset.read(1, out_shape=(output_size, output_size), resampling=Resampling.bilinear).astype(np.float32))
        return np.stack(arrays)

    optical = read_group(paths["sentinel_2"], optical_bands)
    sar = read_group(paths["sentinel_1"], sar_bands)
    return optical, sar
