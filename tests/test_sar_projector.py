import pytest
import torch
from src.eo_vlm.sar_projector import S1SARProjector
def test_sar_projector_shape_and_finite_output():
 p=S1SARProjector();out=p(torch.randn(2,225,768));assert out.shape==(2,16,2048) and torch.isfinite(out).all()
def test_sar_projector_rejects_bad_or_nonfinite_input():
 p=S1SARProjector()
 with pytest.raises(ValueError):p(torch.randn(1,224,768))
 x=torch.randn(1,225,768);x[0,0,0]=float('nan')
 with pytest.raises(ValueError):p(x)
def test_sar_projector_is_input_sensitive():
 p=S1SARProjector().eval();a=torch.randn(1,225,768);b=a.clone();b[:,0,0]+=1;assert not torch.equal(p(a),p(b))
