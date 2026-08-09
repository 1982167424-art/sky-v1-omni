from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ModalConfig

class VideoTokenizer(nn.Module):
    def __init__(self, cfg: ModalConfig, hidden_size: int):
        super().__init__()
        self.num_frames = int(cfg.num_frames)
        self.ps = int(cfg.patch_size)
        self.fsz = int(cfg.frame_size)
        self.modal_id = int(cfg.modal_id)
        in_dim = 3 * self.ps * self.ps
        self.proj = nn.Linear(in_dim, hidden_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"VideoTokenizer expects 5D tensor (B,T,C,H,W), got {tuple(video.shape)}")
        B, T, C, H, W = video.shape
        if H % self.ps != 0 or W % self.ps != 0:
            nh = (H // self.ps) * self.ps
            nw = (W // self.ps) * self.ps
            video = video[..., :nh, :nw].contiguous()
            _, _, _, H, W = video.shape
        nh, nw = H // self.ps, W // self.ps
        frames = video.view(B * T, C, H, W)
        patches = frames.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * T, nh * nw, -1)
        emb = self.proj(patches)
        emb = emb.view(B, T * nh * nw, -1)
        return emb
