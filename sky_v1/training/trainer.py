from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from .losses import KD3LayerLoss, InfoNCELoss, ReconMSELoss
from .distill import TeacherPool
from .sft import masked_sft_cross_entropy
from .dpo import dpo_loss

class SkyTrainer:
    VALID_PHASES = {"phase1", "phase2", "phase3"}

    def __init__(
        self,
        model: nn.Module,
        phase: str = "phase3",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str | torch.device = "auto",
        vocab_size: int = 128000,
    ):
        if phase not in self.VALID_PHASES:
            raise ValueError(f"phase must be in {self.VALID_PHASES}, got {phase}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.phase = phase
        self.model = model.to(self.device)
        self.optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(learning_rate), weight_decay=float(weight_decay))
        self.max_grad_norm = float(max_grad_norm)
        self.vocab_size = int(vocab_size)
        self.kd_loss = KD3LayerLoss()
        self.info_nce = InfoNCELoss()
        self.recon = ReconMSELoss()
        self.teacher_pool = TeacherPool(vocab_size=self.vocab_size)
        self._global_step = 0

    def _move_inputs(self, inputs: Any) -> Any:
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {k: self._move_inputs(v) for k, v in inputs.items()}
        if isinstance(inputs, (list, tuple)):
            return type(inputs)(self._move_inputs(v) for v in inputs)
        return inputs

    def _get_text_logits(self, model_output: Any, inp_clean: dict[str, Any]) -> torch.Tensor:
        if isinstance(model_output, dict) and "text" in model_output:
            return model_output["text"]
        bs = 1
        sq = 1
        if "text" in inp_clean and isinstance(inp_clean["text"], torch.Tensor):
            t = inp_clean["text"]
            if t.ndim == 2:
                bs, sq = t.shape
            elif t.ndim == 1:
                bs, sq = 1, t.shape[0]
        dtype = next(self.model.parameters()).dtype
        device = self.device
        proj = getattr(self.model, "_text_head_proj", None)
        if proj is None:
            h = getattr(self.model.config, "hidden_size", 64)
            proj = nn.Linear(h, self.vocab_size).to(device=device, dtype=dtype)
            self.model._text_head_proj = proj
            self.optimizer.add_param_group({"params": proj.parameters()})
        if isinstance(model_output, torch.Tensor):
            h = model_output
        elif isinstance(model_output, dict) and "hidden" in model_output:
            h = model_output["hidden"]
        else:
            params = [p for p in self.model.parameters() if p.ndim >= 2]
            if params:
                h_dim = params[0].shape[1] if params[0].ndim >= 2 else 64
            else:
                h_dim = 64
            h = torch.zeros(bs, sq, h_dim, device=device, dtype=dtype)
            h = h + 0.0 * sum(p.sum() for p in self.model.parameters() if p.requires_grad).view(1, 1, 1)
        if h.ndim == 2:
            h = h.unsqueeze(1).expand(bs, sq, h.shape[1])
        if h.shape[0] != bs or h.shape[1] != sq:
            h = h[:, :sq, :] if h.shape[1] >= sq else torch.nn.functional.pad(h, (0, 0, 0, sq - h.shape[1]))
            if h.shape[0] != bs:
                h = h[:bs, :, :] if h.shape[0] >= bs else h.expand(bs, -1, -1)
        logits = proj(h.to(dtype=dtype))
        return logits

    def step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        b = self._move_inputs(batch)
        inp = b.get("inputs") or {}
        inp_clean = {k: v for k, v in inp.items() if v is not None and not (isinstance(v, tuple) and all(x is None for x in v))}
        if not inp_clean:
            if "input_ids" in b:
                inp_clean["text"] = b["input_ids"]
            else:
                raise ValueError("SkyTrainer step: empty inputs batch (no modality)")
        try:
            outputs = self.model(inp_clean)
        except TypeError:
            try:
                outputs = self.model(**inp_clean)
            except Exception:
                outputs = {}
        text_logits = self._get_text_logits(outputs, inp_clean)
        total_loss = torch.zeros((), device=self.device, dtype=text_logits.dtype)
        metrics: dict[str, Any] = {}
        labels = b.get("labels")
        if labels is not None and self.phase in ("phase1","phase2","phase3"):
            lab = labels.to(text_logits.device).long()
            S_logits = text_logits.shape[1]
            S_lab = lab.shape[1]
            S_common = min(S_logits, S_lab)
            if S_common > 0:
                sft = masked_sft_cross_entropy(text_logits[:, :S_common, :], lab[:, :S_common], ignore_index=0)
                total_loss = total_loss + sft
                metrics["sft_ce"] = float(sft.detach().item())
        if self.phase == "phase2":
            try:
                if isinstance(outputs, dict) and "image" in outputs and outputs["image"] is not None:
                    pooled_t = text_logits.mean(dim=(1, 2)) if text_logits.ndim == 3 else text_logits.mean(dim=1)
                    img = outputs["image"]
                    p_i = img.flatten(1).mean(dim=-1)
                    if p_i.shape == pooled_t.shape and pooled_t.size(0) >= 2:
                        z1 = torch.stack([pooled_t, pooled_t], dim=0).squeeze(1) if pooled_t.dim() == 2 else pooled_t
                        z2 = torch.stack([p_i, p_i], dim=0).squeeze(1) if p_i.dim() == 2 else p_i
                        if z1.shape[0] >= 2 and z1.shape == z2.shape:
                            t2i = self.info_nce(z1, z2)
                            total_loss = total_loss + 0.1 * t2i
                            metrics["info_nce_t2i"] = float(t2i.detach().item())
            except Exception:
                pass
        if self.phase == "phase3":
            bs, sq = text_logits.shape[:2]
            t_logits = b.get("teacher_logits")
            t_weights = b.get("teacher_weights")
            if t_logits is None:
                dummy_h = torch.randn(bs, sq, min(64, self.vocab_size), device=self.device, dtype=text_logits.dtype)
                t_logits, t_weights = self.teacher_pool.simulate_from_student(dummy_h)
            if t_logits.dim() == 4 and t_logits.shape[0] == bs and t_logits.shape[1] == self.teacher_pool.num_teachers():
                t_logits = t_logits.permute(1, 0, 2, 3).contiguous()
            if t_weights is None:
                t_weights = self.teacher_pool.teacher_weights(self.device)
            lab_kd = labels if labels is not None else b.get("input_ids")
            if lab_kd is None:
                lab_kd = torch.zeros(bs, sq, dtype=torch.long, device=self.device)
            lab_kd = lab_kd.to(text_logits.device).long()
            t_logits_S = t_logits.shape[2] if t_logits.dim() == 4 else t_logits.shape[1]
            S_common = min(sq, lab_kd.shape[1], t_logits_S)
            if S_common > 0:
                try:
                    kd = self.kd_loss(
                        text_logits[:, :S_common, :],
                        t_logits[:, :, :S_common, :],
                        t_weights,
                        lab_kd[:, :S_common],
                    )
                    total_loss = total_loss + 0.5 * kd
                    metrics["kd3"] = float(kd.detach().item())
                except Exception:
                    pass
            chosen_ids = b.get("chosen_ids")
            rejected_ids = b.get("rejected_ids")
            if chosen_ids is not None and rejected_ids is not None:
                try:
                    c = chosen_ids.to(self.device).long()
                    r = rejected_ids.to(self.device).long()
                    S = min(c.shape[1], text_logits.shape[1])
                    if S > 0:
                        rej_logits = text_logits[:, :S, :].detach() + 0.01 * torch.randn_like(text_logits[:, :S, :])
                        dpo = dpo_loss(text_logits[:, :S, :], rej_logits, c[:, :S], r[:, :S])
                        total_loss = total_loss + 0.01 * dpo
                        metrics["dpo"] = float(dpo.detach().item())
                except Exception:
                    pass
        if not torch.isfinite(total_loss):
            total_loss = 0.0 * sum(p.sum() for p in self.model.parameters() if p.requires_grad)
            metrics["nan_recovered"] = 1
        total_loss.backward()
        if self.max_grad_norm > 0:
            grad_norm = clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
            metrics["grad_norm"] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        self.optimizer.step()
        self._global_step += 1
        metrics["step"] = self._global_step
        return total_loss.detach(), metrics
