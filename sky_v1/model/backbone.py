from __future__ import annotations
import torch
import torch.nn as nn
from .config import SkyModelConfig
from .norm import RMSNorm
from .embeddings import RotaryPositionalEmbedding
from .transformer_layer import UniTransformerLayer

class UniTransformerBackbone(nn.Module):
    def __init__(self, config: SkyModelConfig):
        super().__init__()
        self.config = config
        self.rope = RotaryPositionalEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            UniTransformerLayer(config, layer_idx=i, rope=self.rope)
            for i in range(config.num_hidden_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Backbone expects 3D tensor (B,S,H), got shape={tuple(hidden_states.shape)}")
        _, S, _ = hidden_states.shape
        if S > self.config.max_position_embeddings:
            raise ValueError(f"seq_len {S} > max_position_embeddings {self.config.max_position_embeddings}")
        x = hidden_states
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.final_norm(x)
