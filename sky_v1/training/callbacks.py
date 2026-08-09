from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

class MetricsLogger:
    def __init__(self, log_dir: str | Path):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"metrics_{int(time.time())}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def log(self, **metrics: Any) -> None:
        metrics.setdefault("ts", time.time())
        self._f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try: self._f.close()
        except Exception: pass
    def __del__(self):
        try: self.close()
        except Exception: pass
