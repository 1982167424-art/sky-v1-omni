from __future__ import annotations
import json
import math
import shutil
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

class CheckpointManager:
    def __init__(self, output_dir: str | Path, keep_last_k: int = 5):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = int(keep_last_k)
        self._history: list[dict[str, Any]] = []
        self._best: dict[str, Any] | None = None
        self._index_path = self.dir / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._history = list(data.get("history", []))
                self._best = data.get("best")
            except Exception:
                pass

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps({"history": self._history, "best": self._best}, indent=2), encoding="utf-8")

    def best_state(self) -> dict[str, Any] | None: return self._best

    def on_step_end(self, step: int, model: nn.Module, optimizer: Any, loss: float) -> Path:
        step = int(step)
        loss = float(loss) if loss is not None else float("inf")
        is_nan = (not math.isfinite(loss))
        path = self.dir / f"checkpoint_step_{step:06d}.pt"
        save = {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        }
        torch.save(save, path)
        if is_nan:
            self.rollback_last_best(model, optimizer)
            return path
        entry = {"step": step, "loss": loss, "path": str(path)}
        self._history.append(entry)
        self._history.sort(key=lambda e: e["step"])
        if self._best is None or loss < self._best["loss"]:
            self._best = dict(entry)
            shutil.copyfile(path, self.dir / "best.pt")
            self._best["path"] = str(self.dir / "best.pt")
        if len(self._history) > self.keep_last_k:
            old = self._history.pop(0)
            try: Path(old["path"]).unlink(missing_ok=True)
            except Exception: pass
        self._save_index()
        return path

    def rollback_last_best(self, model: nn.Module, optimizer: Any) -> None:
        if self._best is None:
            return
        p = Path(self._best["path"])
        if not p.exists():
            return
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
