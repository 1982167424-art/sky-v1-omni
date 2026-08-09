from __future__ import annotations
import torch
import torch.nn as nn
from ..config import ModalConfig

class TextTokenizer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, modal_id: int = 0):
        super().__init__()
        if vocab_size < 1 or hidden_size < 1:
            raise ValueError("sizes must be positive")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.modal_id = int(modal_id)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("TextTokenizer input_ids must be integer tensor")
        if input_ids.ndim != 2:
            raise ValueError(f"TextTokenizer expects 2D ids (B,S), got {tuple(input_ids.shape)}")
        if (input_ids < 0).any() or (input_ids >= self.vocab_size).any():
            input_ids = input_ids.clamp(0, self.vocab_size - 1)
        return self.embedding(input_ids)
