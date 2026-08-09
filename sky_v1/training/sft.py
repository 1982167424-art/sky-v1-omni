from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_sft_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B,S,V), got {tuple(logits.shape)}")
    B, S, V = logits.shape
    if labels.shape != (B, S):
        raise ValueError(f"labels must be (B,S), got {tuple(labels.shape)}")
    return F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=ignore_index)
