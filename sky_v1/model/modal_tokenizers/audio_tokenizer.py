from __future__ import annotations
import torch
import torch.nn as nn
from ..config import ModalConfig

class AudioTokenizer(nn.Module):
    def __init__(self, cfg: ModalConfig, hidden_size: int):
        super().__init__()
        self.mel_bins = int(cfg.mel_bins)
        self.sub = int(cfg.subsample)
        self.modal_id = int(cfg.modal_id)
        ks = self.sub * 2
        self.conv = nn.Conv1d(self.mel_bins, hidden_size, kernel_size=ks, stride=self.sub, padding=ks // 2)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"AudioTokenizer expects 3D tensor (B,mel,T), got {tuple(mel.shape)}")
        out = self.conv(mel)
        return out.transpose(1, 2).contiguous()
