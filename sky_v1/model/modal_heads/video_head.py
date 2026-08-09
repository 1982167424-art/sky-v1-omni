from __future__ import annotations
import torch
import torch.nn as nn
from ..config import HeadsConfig

class VideoHead(nn.Module):
    def __init__(self, cfg: HeadsConfig, hidden_size: int, frame_size: int = 224):
        super().__init__()
        self.num_frames = int(cfg.num_frames)
        self.ps = int(cfg.patch_size)
        self.out_ch = int(cfg.out_channels)
        self.fsz = int(frame_size)
        if self.fsz % self.ps != 0:
            raise ValueError("frame_size must divide by patch_size")
        self.nh = self.fsz // self.ps
        self.tokens_per_frame = self.nh * self.nh
        self.proj = nn.Linear(hidden_size, self.out_ch * self.ps * self.ps)
        self.shuffle = nn.PixelShuffle(self.ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        need = self.num_frames * self.tokens_per_frame
        if S < need:
            x = torch.nn.functional.pad(x, (0, 0, 0, need - S))
        elif S > need:
            x = x[:, :need]
        x = x.view(B, self.num_frames, self.tokens_per_frame, -1)
        flat = self.proj(x).view(B, self.num_frames, self.nh, self.nh, self.out_ch, self.ps, self.ps)
        flat = flat.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        frames = flat.view(B, self.num_frames, self.out_ch, self.fsz, self.fsz)
        return frames
