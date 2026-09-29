"""Standalone CROMA-SAR-token to Qwen visual-token adapter; no fusion."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .multispectral_projector import QWEN_HIDDEN_SIZE, QWEN_IMAGE_TOKEN_COUNT

class S1SARProjector(nn.Module):
    """Map CROMA S1 tokens [B,225,768] to Qwen's [B,16,2048] interface."""
    input_tokens=225; input_hidden_size=768; token_count=QWEN_IMAGE_TOKEN_COUNT; output_hidden_size=QWEN_HIDDEN_SIZE
    def __init__(self) -> None:
        super().__init__(); self.norm=nn.LayerNorm(768);self.output=nn.Sequential(nn.Linear(768,2048),nn.GELU(),nn.Linear(2048,2048))
    def forward(self, sar_tokens:torch.Tensor)->torch.Tensor:
        if sar_tokens.ndim!=3 or tuple(sar_tokens.shape[1:])!=(225,768):raise ValueError('SAR CROMA tokens must be [batch,225,768]')
        if not torch.isfinite(sar_tokens).all():raise ValueError('SAR CROMA tokens contain non-finite values')
        b=sar_tokens.shape[0];grid=sar_tokens.transpose(1,2).reshape(b,768,15,15);pooled=F.adaptive_avg_pool2d(grid,(4,4)).flatten(2).transpose(1,2)
        return self.output(self.norm(pooled))
