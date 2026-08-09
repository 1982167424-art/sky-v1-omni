from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KD3LayerLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, temperature: float = 2.0):
        super().__init__()
        if abs(alpha + beta + gamma) < 1e-9:
            raise ValueError("alpha+beta+gamma must be > 0")
        s = alpha + beta + gamma
        self.alpha = float(alpha / s)
        self.beta = float(beta / s)
        self.gamma = float(gamma / s)
        self.T = float(temperature)
        if self.T <= 0: raise ValueError("T must be >0")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_weights: torch.Tensor,
        labels: torch.Tensor,
        hidden_student: torch.Tensor | None = None,
        hidden_teachers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, V = student_logits.shape
        T_t = teacher_logits.size(0)
        if teacher_weights.numel() != T_t:
            raise ValueError(f"teacher_weights length {teacher_weights.numel()} != num teachers {T_t}")
        w = teacher_weights.to(dtype=student_logits.dtype).clamp(min=0.0)
        if w.sum() <= 0:
            w = torch.ones_like(w)
        w = w / w.sum()
        tp = teacher_logits.to(dtype=student_logits.dtype)
        tp = tp - tp.logsumexp(dim=-1, keepdim=True)
        tp = torch.exp(tp)
        weighted = torch.einsum("tbsv,t->bsv", tp, w)
        student_logprobs = F.log_softmax(student_logits / self.T, dim=-1)
        kl = F.kl_div(student_logprobs, weighted, reduction="batchmean") * (self.T ** 2)
        ce = F.cross_entropy(student_logits.view(-1, V), labels.view(-1), ignore_index=-100)
        if hidden_student is not None and hidden_teachers is not None:
            hm = torch.einsum("tbhw...,t->bhw...", hidden_teachers.to(dtype=student_logits.dtype), w)
            mse = F.mse_loss(hidden_student.float(), hm.float())
        else:
            mse = torch.zeros((), dtype=student_logits.dtype, device=student_logits.device)
        return self.alpha * kl + self.beta * ce + self.gamma * mse

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = float(temperature)
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"InfoNCE shape mismatch z1={tuple(z1.shape)} z2={tuple(z2.shape)}")
        z1 = F.normalize(z1.float(), dim=-1, p=2)
        z2 = F.normalize(z2.float(), dim=-1, p=2)
        N = z1.size(0)
        logits = z1 @ z2.T / self.T
        labels = torch.arange(N, device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

class ReconMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred.float(), target.float())
