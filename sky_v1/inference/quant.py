"""sky_v1.inference.quant: simple W8A8 / W4A16 (fp16 weight + 4bit scale) quant layers.

These are pedagogical/small-production implementations, not kernel-fused CUDA ops.
For production, replace with AutoGPTQ/AWQ/bitsandbytes while keeping the
`quantize_model_` / `dequantize_model_` public interface.
"""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_group_quantize(
    w: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w.shape
    num_groups = (in_dim + group_size - 1) // group_size
    pad = num_groups * group_size - in_dim
    w_pad = F.pad(w.float(), (0, pad)) if pad > 0 else w.float()
    groups = w_pad.view(out_dim, num_groups, group_size)
    absmax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    qmax = (1 << (bits - 1)) - 1
    scales = absmax / qmax
    q = (groups / scales).round().clamp(-qmax, qmax).to(torch.int8)
    q = q.view(out_dim, num_groups * group_size)[:, :in_dim].contiguous()
    scales = scales.squeeze(-1)
    return q, scales


def _per_group_dequantize(
    q: torch.Tensor, scales: torch.Tensor, group_size: int, orig_in: int
) -> torch.Tensor:
    out_dim, in_dim = q.shape
    num_groups = (in_dim + group_size - 1) // group_size
    pad = num_groups * group_size - in_dim
    q_pad = F.pad(q.float(), (0, pad)) if pad > 0 else q.float()
    groups = q_pad.view(out_dim, num_groups, group_size)
    s = scales.unsqueeze(-1)
    dq = (groups * s).view(out_dim, num_groups * group_size)[:, :orig_in]
    return dq


class W8A8Linear(nn.Module):
    __constants__ = ("in_features", "out_features", "group_size", "bits")

    def __init__(
        self, in_features: int, out_features: int, group_size: int = 128, bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = 8
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.zeros(out_features, (in_features + group_size - 1) // group_size, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32) if bias else torch.zeros(0))

    @classmethod
    def from_float(cls, mod: nn.Linear, group_size: int = 128) -> "W8A8Linear":
        assert isinstance(mod, nn.Linear)
        qlayer = cls(mod.in_features, mod.out_features, group_size=group_size, bias=mod.bias is not None)
        q, s = _per_group_quantize(mod.weight.detach().cpu(), 8, group_size)
        qlayer.qweight.copy_(q)
        qlayer.scales.copy_(s)
        if mod.bias is not None:
            qlayer.bias.copy_(mod.bias.detach().cpu().float())
        return qlayer

    def to_float(self) -> nn.Linear:
        w = _per_group_dequantize(self.qweight, self.scales, self.group_size, self.in_features)
        linear = nn.Linear(self.in_features, self.out_features, bias=(self.bias.numel() > 0))
        linear.weight.data.copy_(w.to(linear.weight.dtype))
        if self.bias.numel() > 0:
            linear.bias.data.copy_(self.bias.to(linear.bias.dtype))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _per_group_dequantize(self.qweight, self.scales, self.group_size, self.in_features)
        w = w.to(x.dtype).to(x.device)
        y = F.linear(x, w)
        if self.bias.numel() > 0:
            y = y + self.bias.to(y.dtype).to(y.device)
        return y


class W4A16Linear(nn.Module):
    __constants__ = ("in_features", "out_features", "group_size", "bits")

    def __init__(
        self, in_features: int, out_features: int, group_size: int = 128, bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = 4
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.zeros(out_features, (in_features + group_size - 1) // group_size, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32) if bias else torch.zeros(0))

    @classmethod
    def from_float(cls, mod: nn.Linear, group_size: int = 128) -> "W4A16Linear":
        qlayer = cls(mod.in_features, mod.out_features, group_size=group_size, bias=mod.bias is not None)
        q, s = _per_group_quantize(mod.weight.detach().cpu(), 4, group_size)
        qlayer.qweight.copy_(q)
        qlayer.scales.copy_(s)
        if mod.bias is not None:
            qlayer.bias.copy_(mod.bias.detach().cpu().float())
        return qlayer

    def to_float(self) -> nn.Linear:
        w = _per_group_dequantize(self.qweight, self.scales, self.group_size, self.in_features)
        linear = nn.Linear(self.in_features, self.out_features, bias=(self.bias.numel() > 0))
        linear.weight.data.copy_(w.to(linear.weight.dtype))
        if self.bias.numel() > 0:
            linear.bias.data.copy_(self.bias.to(linear.bias.dtype))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _per_group_dequantize(self.qweight, self.scales, self.group_size, self.in_features)
        w = w.to(x.dtype).to(x.device)
        y = F.linear(x, w)
        if self.bias.numel() > 0:
            y = y + self.bias.to(y.dtype).to(y.device)
        return y


_LINEAR_CLS_MAP = {
    "w8a8": W8A8Linear,
    "gptq": W8A8Linear,
    "awq": W4A16Linear,
    "w4a16": W4A16Linear,
    "bnb": W4A16Linear,
}


def quantize_model_(model: nn.Module, quant_config: dict[str, Any]) -> None:
    mode = (quant_config or {}).get("mode", "none") or "none"
    if mode == "none":
        return
    Cls = _LINEAR_CLS_MAP.get(mode)
    if Cls is None:
        raise ValueError(f"Unknown quant mode: {mode}. Use one of {list(_LINEAR_CLS_MAP)}")
    group_size = int((quant_config or {}).get("group_size", 128))

    def _should_replace(name: str, mod: nn.Module) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        targets = (quant_config or {}).get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj", "ffn", "proj", "lm_head", "head"],
        )
        return any(t.lower() in name.lower() for t in targets) or True

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if _should_replace(full, child) and isinstance(child, nn.Linear):
                setattr(parent, child_name, Cls.from_float(child, group_size=group_size))


def dequantize_model_(model: nn.Module) -> None:
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, (W8A8Linear, W4A16Linear)):
                setattr(parent, child_name, child.to_float())
