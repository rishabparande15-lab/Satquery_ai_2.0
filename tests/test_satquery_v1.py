import numpy as np
from src.satquery_v1 import SatQueryV1Controller

def metadata():return {'crs':'EPSG:32631','resolution':[10.,10.],'bounds':[0.,0.,1200.,1200.]}
def test_primary_route_requires_both_modalities_without_loading_models():
 c=SatQueryV1Controller(device='cpu');common={'task_type':'caption','question':'describe','patch_id':'p','s1_patch_id':'p','s2_patch_id':'p','spatial_metadata':metadata()}
 assert c.run_satquery(s1=None,s2=np.zeros((12,120,120),np.float32),**common)['error_code']=='MISSING_S1_FOR_MULTIMODAL_ROUTE'
 assert c.run_satquery(s1=np.zeros((2,120,120),np.float32),s2=None,**common)['error_code']=='MISSING_S2_FOR_MULTIMODAL_ROUTE'
def test_primary_route_rejects_identity_and_shape():
 c=SatQueryV1Controller(device='cpu');base={'task_type':'caption','question':'describe','patch_id':'p','s1':np.zeros((2,120,120),np.float32),'s2':np.zeros((12,120,120),np.float32),'s1_patch_id':'p','s2_patch_id':'wrong','spatial_metadata':metadata()}
 assert c.run_satquery(**base)['error_code']=='MISMATCHED_PATCH_ID'

def test_controller_uses_standard_croma_environment_variables(monkeypatch):
    monkeypatch.setenv('CROMA_SOURCE', r'D:\Satquery_ai datasets\croma_official')
    monkeypatch.setenv('CROMA_CHECKPOINT', r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt')
    monkeypatch.delenv('SATQUERY_CROMA_SOURCE', raising=False)
    monkeypatch.delenv('SATQUERY_CROMA_CHECKPOINT', raising=False)
    controller = SatQueryV1Controller(device='cpu')
    assert str(controller.croma_source) == r'D:\Satquery_ai datasets\croma_official'
    assert str(controller.croma_checkpoint) == r'D:\Satquery_ai datasets\checkpoints\CROMA_base.pt'
