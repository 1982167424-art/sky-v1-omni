from __future__ import annotations
import torch
import torch.nn as nn
from ..config import HeadsConfig

class ThreeDHead(nn.Module):
    def __init__(self, cfg: HeadsConfig, hidden_size: int):
        super().__init__()
        self.n_pts = int(cfg.num_points)
        self.p_dim = int(cfg.point_dim)
        self.n_mv = int(cfg.mesh_vertices)
        self.point_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, self.n_pts * self.p_dim),
        )
        self.mesh_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, self.n_mv * 3),
        )
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = x.mean(dim=1)
        pts = self.point_head(pooled).view(x.size(0), self.n_pts, self.p_dim)
        mv = self.mesh_head(pooled).view(x.size(0), self.n_mv, 3)
        return pts, mv
