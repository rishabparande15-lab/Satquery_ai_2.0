import numpy as np
from src.pixel_features import aggregate_tokens, candidate_maps, local_mean_std, safe_normalized_difference


def test_normalized_difference_masks_bad_denominators():
    out, valid = safe_normalized_difference(np.array([3, 1, np.nan]), np.array([1, -1, 2]))
    assert out[0] == .5 and valid.tolist() == [True, False, False]
    assert np.isnan(out[1:]).all()


def test_candidate_formulas_and_finite_inputs():
    optical = np.ones((12, 120, 120), dtype=np.float32)
    sar = np.stack((np.full((120, 120), -5), np.full((120, 120), -12))).astype(np.float32)
    order = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
    maps = candidate_maps(optical, sar, order, ("VV", "VH"))
    assert maps["NDVI"].shape == (120, 120) and np.all(maps["NDVI"] == 0)
    assert np.all(maps["VV_minus_VH"] == 7)


def test_token_alignment_is_row_major_and_exact():
    values = np.repeat(np.arange(15, dtype=np.float32), 8)[:, None] * 100
    values = values + np.repeat(np.arange(15, dtype=np.float32), 8)[None, :]
    tokens, names = aggregate_tokens(values)
    assert names[-1] == "valid_ratio" and tokens.shape == (225, 8)
    assert tokens[0, 0] == 0 and tokens[1, 0] == 1 and tokens[15, 0] == 100 and np.all(tokens[:, -1] == 1)


def test_local_window_shape_reproducibility_and_constant_std():
    x = np.ones((120, 120), dtype=np.float32)
    for window in (3, 5, 7):
        mean, std = local_mean_std(x, window)
        assert mean.shape == x.shape and std.shape == x.shape
        assert np.array_equal(mean, local_mean_std(x, window)[0])
        assert np.all(std == 0)
