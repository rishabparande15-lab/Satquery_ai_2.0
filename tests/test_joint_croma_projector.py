import pytest,torch
from src.eo_vlm.joint_croma_projector import CromaJointProjector

def test_joint_projector_contract_and_sensitivity():
    p=CromaJointProjector().eval();a=torch.randn(1,225,768);b=a.clone();b[:,0,0]+=1
    assert p(a).shape==(1,16,2048) and torch.isfinite(p(a)).all() and not torch.equal(p(a),p(b))

def test_joint_projector_fails_closed_on_invalid_inputs():
    p=CromaJointProjector()
    with pytest.raises(ValueError):p(torch.randn(1,224,768))
    a=torch.randn(1,225,768);a[0,0,0]=float('nan')
    with pytest.raises(ValueError):p(a)
