from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import Any

_PHASES = {"text", "image", "audio", "video", "three_d", "all"}

class Phase1Dataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], phase: str = "text"):
        if phase not in _PHASES:
            raise ValueError(f"phase must be in {_PHASES}, got {phase}")
        self.samples = list(samples)
        self.phase = phase
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        out: dict[str, Any] = {"id": s.get("id", f"s{idx}"), "phase": self.phase}
        if self.phase in ("text", "all"):
            out.update(input_ids=s["text_ids"].clone(), labels=s["text_labels"].clone())
        if self.phase in ("image", "all"):
            out["image"] = s["image"].clone()
        if self.phase in ("audio", "all"):
            out["audio"] = s["audio"].clone()
        if self.phase in ("video", "all"):
            out["video"] = s["video"].clone()
        if self.phase in ("three_d", "all"):
            out["three_d_points"] = s["three_d_points"].clone()
            out["three_d_mesh"] = s["three_d_mesh"].clone()
        return out

class Phase2AlignDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = list(samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        return {
            "id": s.get("id", f"s{idx}"),
            "input_ids": s["text_ids"].clone(),
            "labels": s["text_labels"].clone(),
            "image": s["image"].clone(),
            "audio": s["audio"].clone(),
            "video": s["video"].clone(),
            "three_d_points": s["three_d_points"].clone(),
            "three_d_mesh": s["three_d_mesh"].clone(),
        }

class Phase3DistillDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], vocab_size: int = 1000, num_teachers: int = 5):
        self.samples = list(samples)
        self.vocab_size = int(vocab_size)
        self.num_teachers = int(num_teachers)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        seq = s["text_ids"].numel()
        t = torch.randn(self.num_teachers, seq, self.vocab_size, dtype=torch.float32)
        t = torch.softmax(t, dim=-1)
        chosen = s["text_ids"].clone()
        reject = torch.randint_like(chosen, low=1, high=self.vocab_size)
        return {
            "id": s.get("id", f"s{idx}"),
            "input_ids": s["text_ids"].clone(),
            "labels": s["text_labels"].clone(),
            "image": s["image"].clone(),
            "audio": s["audio"].clone(),
            "video": s["video"].clone(),
            "three_d_points": s["three_d_points"].clone(),
            "three_d_mesh": s["three_d_mesh"].clone(),
            "teacher_logits": t,
            "teacher_weights": torch.tensor([1.2, 1.3, 1.4, 1.2, 1.0], dtype=torch.float32),
            "chosen_ids": chosen,
            "rejected_ids": reject,
        }
