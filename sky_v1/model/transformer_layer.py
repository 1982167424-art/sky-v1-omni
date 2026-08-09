from __future__ import annotations
import torch
import torch.nn as nn
from .config import SkyModelConfig
from .norm import RMSNorm
from .attention import MultiHeadAttention
from .ffn import SwiGLUFFN
from .embeddings import RotaryPositionalEmbedding

class UniTransformerLayer(nn.Module):
    def __init__(self, config: SkyModelConfig, layer_idx: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout_p=config.attention_dropout,
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config.hidden_size, config.intermediate_size, act=config.hidden_act)
        self.rope = rope

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = self.attn_norm(x)
        a = self.attn(n, n, n, attn_mask=attn_mask, rope=self.rope)
        x = x + a
        n2 = self.ffn_norm(x)
        f = self.ffn(n2)
        return x + f
