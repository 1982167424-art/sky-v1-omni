from __future__ import annotations
import torch
import torch.nn as nn

class TeacherPool(nn.Module):
    def __init__(self, vocab_size: int = 1000, teacher_names: tuple[str, ...] = (
        "claude_opus_4_8", "gpt_5_6_sol", "kimi_k3", "mimo_v2_5", "qwen_3_8",
    )):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.teacher_names = tuple(teacher_names)
        self._projs = nn.ModuleList([nn.Linear(64, self.vocab_size) for _ in self.teacher_names])

    def num_teachers(self) -> int: return len(self.teacher_names)

    def teacher_weights(self, device: torch.device) -> torch.Tensor:
        w = [1.2, 1.3, 1.4, 1.2, 1.0]
        if len(w) != len(self.teacher_names):
            w = [1.0] * len(self.teacher_names)
        return torch.tensor(w, device=device, dtype=torch.float32)

    def simulate_from_student(self, student_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = student_hidden.shape
        device = student_hidden.device
        gen = torch.Generator(device=device).manual_seed(1234)
        mat = torch.randn(H, 64, generator=gen, device=device, dtype=student_hidden.dtype) / (H ** 0.5)
        reduced = student_hidden @ mat
        outs: list[torch.Tensor] = []
        for p in self._projs:
            outs.append(p(reduced))
        logits = torch.stack(outs, dim=0)
        return logits, self.teacher_weights(device)
