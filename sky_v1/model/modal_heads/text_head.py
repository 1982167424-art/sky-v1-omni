from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import HeadsConfig

class TextHead(nn.Module):
    def __init__(self, cfg: HeadsConfig, hidden_size: int):
        super().__init__()
        self.vocab_size = int(cfg.vocab_size)
        self.hidden_size = int(hidden_size)
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)
    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / float(temperature)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, mask)
            logits = logits.masked_fill(remove, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(*probs.shape[:-1])
