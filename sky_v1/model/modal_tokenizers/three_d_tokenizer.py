from __future__ import annotations
import torch
import torch.nn as nn
from ..config import ModalConfig

class ThreeDTokenizer(nn.Module):
    def __init__(self, cfg: ModalConfig, hidden_size: int):
        super().__init__()
        self.n_pts = int(cfg.num_points)
        self.pt_dim = int(cfg.point_dim)
        self.n_mv = int(cfg.mesh_vertices)
        self.modal_id = int(cfg.modal_id)
        self.point_enc = nn.Sequential(
            nn.Linear(self.pt_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mesh_enc = nn.Sequential(
            nn.Linear(3, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.point_pool = nn.AdaptiveAvgPool1d(128)
        self.mesh_pool = nn.AdaptiveAvgPool1d(128)

    def forward(self, points: torch.Tensor, mesh_vertices: torch.Tensor | None = None) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"ThreeDTokenizer points expects (B,N,D), got {tuple(points.shape)}")
        p = self.point_enc(points).transpose(1, 2)
        p = self.point_pool(p).transpose(1, 2)
        if mesh_vertices is None:
            mesh_vertices = torch.zeros(points.size(0), self.n_mv, 3, device=points.device, dtype=points.dtype)
        mv = self.mesh_enc(mesh_vertices).transpose(1, 2)
        mv = self.mesh_pool(mv).transpose(1, 2)
        return torch.cat([p, mv], dim=1)
