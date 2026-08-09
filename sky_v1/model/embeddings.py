from __future__ import annotations
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even: {dim}")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.theta = float(theta)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, torch.device("cpu"))

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        if x.ndim != 4:
            raise ValueError(f"RoPE expects 4D tensor (B,H,S,D), got shape={tuple(x.shape)}")
        B, H, S, D = x.shape
        if D != self.dim:
            raise ValueError(f"RoPE D mismatch: x has {D}, expected {self.dim}")
        device = x.device
        cache_needs_rebuild = (
            getattr(self, "cos_cached", None) is None
            or self.cos_cached.shape[0] < S
            or self.cos_cached.device != device
            or self.cos_cached.shape[1] != (self.dim // 2)
        )
        if cache_needs_rebuild:
            self._build_cache(max(S, self.max_seq_len), device)
        cos = self.cos_cached[:S].repeat_interleave(2, dim=-1).view(1, 1, S, D)
        sin = self.sin_cached[:S].repeat_interleave(2, dim=-1).view(1, 1, S, D)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot1 * sin

class ModalTypeEmbedding(nn.Module):
    def __init__(self, num_modal_types: int, hidden_size: int):
        super().__init__()
        self.emb = nn.Embedding(num_modal_types, hidden_size)
    def forward(self, modal_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(modal_ids)
