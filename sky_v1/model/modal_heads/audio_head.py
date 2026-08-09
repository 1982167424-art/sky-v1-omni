from __future__ import annotations
import torch
import torch.nn as nn
from ..config import HeadsConfig

class AudioHead(nn.Module):
    def __init__(self, cfg: HeadsConfig, hidden_size: int):
        super().__init__()
        self.mel_bins = int(cfg.mel_bins)
        self.proj = nn.Linear(hidden_size, self.mel_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.proj(x).transpose(1, 2).contiguous()
        return mel
