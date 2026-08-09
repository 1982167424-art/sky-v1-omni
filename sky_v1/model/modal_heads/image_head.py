from __future__ import annotations
import torch
import torch.nn as nn
from ..config import HeadsConfig

class ImageHead(nn.Module):
    def __init__(self, cfg: HeadsConfig, hidden_size: int, image_size: int = 224):
        super().__init__()
        self.hidden = int(hidden_size)
        self.ps = int(cfg.patch_size)
        self.out_ch = int(cfg.out_channels)
        self.img_sz = int(image_size)
        if self.img_sz % self.ps != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.nh = self.img_sz // self.ps
        self.proj = nn.Linear(self.hidden, self.out_ch * self.ps * self.ps)
        self.shuffle = nn.PixelShuffle(self.ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        expected = self.nh * self.nh
        if N < expected:
            pad = expected - N
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        elif N > expected:
            x = x[:, :expected]
        flat = self.proj(x)
        flat = flat.view(B, self.nh, self.nh, self.out_ch, self.ps, self.ps)
        flat = flat.permute(0, 3, 1, 4, 2, 5).contiguous()
        im = flat.view(B, self.out_ch, self.nh * self.ps, self.nh * self.ps)
        return im
