from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "swiglu"):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("sizes must be positive")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
