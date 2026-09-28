import torch
from src.eo_vlm.training import optimizer_parameter_ids

def test_optimizer_membership_uses_identity_for_differently_shaped_parameters():
    projector = torch.nn.Parameter(torch.zeros(12, 2))
    qwen = torch.nn.Parameter(torch.zeros(2, 2048))
    optimizer = torch.optim.AdamW([projector])
    ids = optimizer_parameter_ids(optimizer)
    assert id(projector) in ids
    assert id(qwen) not in ids
