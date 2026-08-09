from __future__ import annotations
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"RMSNorm dim must be > 0, got {dim}")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            xf = x.float()
        else:
            xf = x
        rms = xf.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = xf * rms
        return (out.to(x.dtype) if x.dtype != out.dtype else out) * self.weight
