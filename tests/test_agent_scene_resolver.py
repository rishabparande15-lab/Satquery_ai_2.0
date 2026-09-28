import pytest
from src.agent.scene_resolver import resolve,SceneResolutionError
def test_fixture_pair_resolves(): assert resolve({'dataset_id':'google/RSRCC','scene_id':'d259f34c_8693_418e_9dab_ef14b860319f','modality':'rgb'})['modality']=='optical_rgb_pair'
def test_bad_scene_fails_closed():
 with pytest.raises(SceneResolutionError,match='SCENE_NOT_FOUND'): resolve({'dataset_id':'x','scene_id':'y','modality':'rgb'})
