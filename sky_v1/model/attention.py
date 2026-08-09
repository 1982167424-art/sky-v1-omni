from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"SDPA expects 4D tensors, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    dtype = q.dtype
    try:
        qk = q @ k.transpose(-2, -1) / math.sqrt(D)
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise ValueError("attn_mask must be bool")
            mask_bc = attn_mask.view(1, 1, Sq, Sk)
            qk = qk.masked_fill(mask_bc, float("-inf"))
        attn = torch.softmax(qk.float(), dim=-1).to(dtype)
        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn = F.dropout(attn, p=dropout_p, training=True)
        return attn @ v
    except Exception:
        qk = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(D)
        if attn_mask is not None:
            mask_bc = attn_mask.view(1, 1, Sq, Sk)
            qk = qk.masked_fill(mask_bc, float("-inf"))
        attn = torch.softmax(qk, dim=-1).to(dtype)
        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn = F.dropout(attn, p=dropout_p, training=True)
        return attn @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) % num_heads ({num_heads}) != 0")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = float(dropout_p)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        B, S, H = q.shape
        qp = self.q_proj(q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        kp = self.k_proj(k).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        vp = self.v_proj(v).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        if rope is not None:
            qp = rope(qp)
            kp = rope(kp)
        attn = scaled_dot_product_attention_safe(qp, kp, vp, attn_mask=attn_mask, dropout_p=self.dropout_p)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(attn)
