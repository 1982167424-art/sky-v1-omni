from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .kv_cache import PagedKVCache

def paged_attention_forward(
    q: torch.Tensor,
    cache: PagedKVCache,
    seq_ids: list[int],
    scale: float | None = None,
) -> torch.Tensor:
    batch, heads, tq, hd = q.shape
    scale = scale or (1.0 / math.sqrt(hd))
    outs = []
    for b in range(batch):
        sid = seq_ids[b]
        K, V, mask = cache.get(sid)
        Tkv = K.shape[2]
        attn = torch.einsum("hqd,hkd->hqk", q[b], K[0]) * scale
        if mask is not None and mask.numel() > 0:
            m = mask[0].unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(m < 0.5, float("-inf"))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        out = torch.einsum("hqk,hkd->hqd", attn, V[0])
        outs.append(out.unsqueeze(0))
    return torch.cat(outs, dim=0)
