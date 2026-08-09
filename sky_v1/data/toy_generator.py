from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Iterator, Any

@dataclass
class ToyDataGenerator:
    n: int = 100
    seed: int = 42
    vocab_size: int = 1000
    text_len: int = 16
    image_size: int = 64
    audio_mel: int = 128
    audio_frames: int = 16
    video_frames: int = 2
    three_d_points: int = 32
    three_d_point_dim: int = 6
    three_d_mesh_verts: int = 16

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int: return self.n

    def generate_one(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        text_ids = rng.integers(1, self.vocab_size, size=(self.text_len,), dtype=np.int64)
        text_labels = np.concatenate([text_ids[1:], np.array([0], dtype=np.int64)])
        image = rng.normal(size=(3, self.image_size, self.image_size)).astype(np.float32)
        audio = rng.normal(size=(self.audio_mel, self.audio_frames)).astype(np.float32)
        video = rng.normal(size=(self.video_frames, 3, self.image_size, self.image_size)).astype(np.float32)
        pts = rng.normal(size=(self.three_d_points, self.three_d_point_dim)).astype(np.float32)
        mesh = rng.normal(size=(self.three_d_mesh_verts, 3)).astype(np.float32)
        return {
            "id": f"toy_{idx}",
            "text_ids": torch.from_numpy(text_ids),
            "text_labels": torch.from_numpy(text_labels),
            "image": torch.from_numpy(image),
            "audio": torch.from_numpy(audio),
            "video": torch.from_numpy(video),
            "three_d_points": torch.from_numpy(pts),
            "three_d_mesh": torch.from_numpy(mesh),
        }

    def generate_all(self) -> Iterator[dict[str, Any]]:
        for i in range(self.n):
            yield self.generate_one(i)
