"""sky_v1.model.lora: Low-Rank Adapters for Linear layers (Hu et al. 2021)."""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen base Linear layer and adds A/B low-rank delta with α/r scaling."""
    __constants__ = ("in_features", "out_features", "rank", "alpha", "scaling")

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(self.dropout(x), self.lora_A)
        delta = F.linear(delta, self.lora_B)
        return base_out + delta.to(base_out.dtype) * self.scaling

    def merge_to_base(self) -> nn.Linear:
        """Bake A/B into base.weight and return plain nn.Linear copy."""
        merged = nn.Linear(
            self.in_features, self.out_features,
            bias=(getattr(self.base, "bias", None) is not None),
        )
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data.copy_((self.base.weight.detach() + delta).to(merged.weight.dtype))
        if getattr(self.base, "bias", None) is not None:
            merged.bias.data.copy_(self.base.bias.detach().to(merged.bias.dtype))
        return merged


def _target_hit(name: str, targets: list[str]) -> bool:
    n = name.lower()
    return any(t.lower() in n for t in targets)


def mark_lora_targets_(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> None:
    """In-place wrap target nn.Linear modules as LoRALinear.

    Default targets: Q/K/V/O projections, FFN gate/up/down, lm_head.
    """
    targets = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up_proj", "down_proj", "ffn",
        "lm_head", "proj", "embed",
    ]
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and _target_hit(full, targets):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))


def merge_lora_(model: nn.Module) -> None:
    """In-place merge all LoRALinear layers back to plain nn.Linear (delta baked)."""
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, child_name, child.merge_to_base())


def unload_lora_(model: nn.Module) -> None:
    """Alias for merge_lora_ for API ergonomics (merge = unload adapter)."""
    merge_lora_(model)


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA params A/B from a model, keyed by full module path."""
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            out[f"{name}.lora_A"] = mod.lora_A.detach().cpu()
            out[f"{name}.lora_B"] = mod.lora_B.detach().cpu()
    return out


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load LoRA params A/B into matching modules (strict key check)."""
    missing = []
    for k, v in state.items():
        if not (k.endswith(".lora_A") or k.endswith(".lora_B")):
            continue
        mod_path, attr = k.rsplit(".", 1)
        mod = model.get_submodule(mod_path) if hasattr(model, "get_submodule") else None
        if mod is None:
            cur: nn.Module = model
            for p in mod_path.split("."):
                if not p:
                    continue
                cur = getattr(cur, p, None)
                if cur is None:
                    break
            mod = cur
        if mod is None or not isinstance(mod, LoRALinear):
            missing.append(k)
            continue
        param = getattr(mod, attr)
        param.data.copy_(v.to(param.dtype).to(param.device))
    if missing:
        raise KeyError(f"Missing LoRA modules for keys: {missing[:3]}")
