from __future__ import annotations
import torch
import torch.nn.functional as F

def dpo_loss(
    logits_chosen: torch.Tensor,
    logits_rejected: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    if logits_chosen.shape[:2] != chosen_ids.shape:
        raise ValueError("logits_chosen shape[:2] must match chosen_ids")
    B, S, V = logits_chosen.shape
    lpc = F.log_softmax(logits_chosen, dim=-1)
    lpr = F.log_softmax(logits_rejected, dim=-1)
    pc = lpc.gather(-1, chosen_ids.clamp(0, V-1).unsqueeze(-1)).squeeze(-1)
    pr = lpr.gather(-1, rejected_ids.clamp(0, V-1).unsqueeze(-1)).squeeze(-1)
    logits_diff = (pc - pr).sum(dim=-1)
    return -F.logsigmoid(torch.tensor(beta, dtype=logits_diff.dtype, device=logits_diff.device) * logits_diff).mean()
