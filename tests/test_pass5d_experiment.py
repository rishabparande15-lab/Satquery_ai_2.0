import numpy as np
from scripts.run_pass5d_experiment import SELECTED, selected_scene_features


def test_selected_pass5d_features_have_frozen_shapes_and_are_finite():
    optical=np.ones((12,120,120),dtype=np.float32); optical[7]=3; optical[3]=1; optical[10]=2
    sar=np.stack((np.full((120,120),-8),np.full((120,120),-14))).astype(np.float32)
    token,scene=selected_scene_features(optical,sar)
    assert SELECTED == ("NDVI","NDWI","NDBI","BSI","VV_minus_VH","NDVI_local_std_5")
    assert token.shape==(225,12) and scene.shape==(24,)
    assert np.isfinite(token).all() and np.isfinite(scene).all()


def test_selected_features_do_not_accept_or_use_targets():
    assert selected_scene_features.__code__.co_varnames[:2] == ("optical","sar")
