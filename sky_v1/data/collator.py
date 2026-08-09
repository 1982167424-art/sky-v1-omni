from __future__ import annotations
import torch
from typing import Any, Sequence

class SkyDataCollator:
    def __init__(self, max_seq_len: int = 256, pad_id: int = 0):
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)

    def _pad1d(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        m = min(max(t.size(0) for t in tensors), self.max_seq_len)
        out = torch.full((len(tensors), m), self.pad_id, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            k = min(t.size(0), m)
            out[i, :k] = t[:k]
        return out

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        items = list(batch)
        out: dict[str, Any] = {}
        keys = set()
        for it in items: keys.update(it.keys())
        for k in keys:
            vals = [it.get(k) for it in items]
            if all(isinstance(v, torch.Tensor) for v in vals):
                shapes = {tuple(v.shape) for v in vals}
                if len(shapes) == 1:
                    out[k] = torch.stack(vals, dim=0)
                elif k in ("input_ids", "labels", "chosen_ids", "rejected_ids") and all(v.ndim == 1 for v in vals):
                    out[k] = self._pad1d(vals)
                else:
                    out[k] = vals
            else:
                out[k] = vals
        out["inputs"] = {
            "text": out.get("input_ids"),
            "image": out.get("image"),
            "audio": out.get("audio"),
            "video": out.get("video"),
            "three_d": (out.get("three_d_points"), out.get("three_d_mesh")) if "three_d_points" in out else None,
        }
        return out
