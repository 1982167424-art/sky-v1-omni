from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ModalConfig

class ImageTokenizer(nn.Module):
    def __init__(self, cfg: ModalConfig, hidden_size: int):
        super().__init__()
        self.ps = int(cfg.patch_size)
        self.img_sz = int(cfg.image_size)
        self.in_ch = int(cfg.in_channels)
        self.hidden = int(hidden_size)
        self.modal_id = int(cfg.modal_id)
        if self.img_sz % self.ps != 0:
            raise ValueError(f"image_size ({self.img_sz}) % patch_size ({self.ps}) != 0")
        in_dim = self.in_ch * self.ps * self.ps
        self.proj = nn.Linear(in_dim, self.hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"ImageTokenizer expects 4D tensor (B,C,H,W), got {tuple(images.shape)}")
        B, C, H, W = images.shape
        if H != self.img_sz or W != self.img_sz:
            images = F.interpolate(images, size=(self.img_sz, self.img_sz), mode="bilinear", align_corners=False)
            B, C, H, W = images.shape
        nh, nw = H // self.ps, W // self.ps
        patches = images.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, nh * nw, -1)
        return self.proj(patches)
