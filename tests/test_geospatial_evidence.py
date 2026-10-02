from src.geospatial_evidence import pair_evidence, raster_evidence


def inspection(**updates):
    base = {"file_name": "input.tif", "sha256": "a" * 64, "driver": "GTiff", "crs": "EPSG:4326", "bounds": [0.0, 0.0, 2.0, 2.0], "transform": [1, 0, 0, 0, -1, 2, 0, 0, 1], "resolution": [1, 1], "width": 2, "height": 2, "band_descriptions": ["B1"], "dtypes": ["float32"], "nodata": [None], "finite_value_checks": {"all_unmasked_values_finite": True}, "band_statistics": [{"band_index": 1, "min": 0.0, "max": 3.0, "mean": 1.5, "std": 1.0, "percentiles": {"p02": 0.0, "p50": 1.5, "p98": 3.0}, "nodata_fraction": 0.0, "finite_fraction": 1.0}]}
    base.update(updates); return base


def test_measured_statistics_and_valid_footprint_are_exposed():
    evidence = raster_evidence("a", inspection())
    assert evidence["raster_statistics"]["label"] == "IMAGE STATISTICS"
    assert evidence["geospatial"]["footprint"]["status"] == "AVAILABLE"
    assert evidence["raster_statistics"]["bands"][0]["mean"] == 1.5


def test_missing_crs_is_not_a_fake_footprint():
    evidence = raster_evidence("rgb", inspection(crs=None, bounds=None))
    assert evidence["geospatial"]["available"] is False
    assert evidence["warnings"] == ["GEOSPATIAL_FOOTPRINT_NOT_AVAILABLE"]


def test_pair_overlap_nonoverlap_and_crs_mismatch_are_fail_closed():
    first = raster_evidence("a", inspection())
    overlap = pair_evidence(first, raster_evidence("b", inspection(bounds=[1, 1, 3, 3])))
    assert overlap["status"] == "OVERLAP"
    assert overlap["coregistration"] == "COREGISTRATION_NOT_VERIFIED"
    assert overlap["pixel_difference"]["status"] == "PIXEL_DIFFERENCE_NOT_AVAILABLE"
    assert pair_evidence(first, raster_evidence("b", inspection(bounds=[3, 3, 4, 4])))["status"] == "NO_OVERLAP"
    assert pair_evidence(first, raster_evidence("b", inspection(crs="EPSG:3857")))["status"] == "PAIR_CRS_MISMATCH"
