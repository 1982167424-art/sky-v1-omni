"""sky_v1.eval.benchmark: unified Benchmark entrypoint.

Implements spec §10.1 Benchmark layer. ``BenchmarkResult`` is a serialisable
dataclass shared by all task evaluators. ``BenchmarkRunner`` dispatches to
per-task evaluators (mmlu / humaneval / throughput) and persists results as JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Container for a single benchmark task's outcome."""

    task_name: str
    accuracy: float
    total_samples: int
    correct: int
    latency_ms_avg: float = 0.0
    throughput_tokens_per_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkResult":
        known = {
            "task_name",
            "accuracy",
            "total_samples",
            "correct",
            "latency_ms_avg",
            "throughput_tokens_per_s",
            "metadata",
        }
        return cls(**{k: v for k, v in d.items() if k in known})


class BenchmarkRunner:
    """Dispatches to per-task evaluators and persists results as JSON.

    ``engine_or_sdk`` accepts either a :class:`SkyInferenceEngine` instance
    (used directly by the evaluators) or a :class:`SkySDK` instance (whose
    underlying engine is unwrapped automatically).
    """

    def __init__(self, engine_or_sdk: Any, device: str = "cpu") -> None:
        self.device = device
        # Unwrap SkySDK -> underlying SkyInferenceEngine so evaluators can call
        # ``engine.chat(messages, ...)`` directly (SDK exposes chat_completions).
        inner = getattr(engine_or_sdk, "_engine", None)
        self.engine = inner if inner is not None else engine_or_sdk
        self._last_results: dict[str, BenchmarkResult] = {}

    def run(self, task_name: str, max_samples: int = 100, **kwargs: Any) -> BenchmarkResult:
        task = task_name.lower().strip()
        if task == "mmlu":
            from .mmlu import eval_mmlu

            result = eval_mmlu(self.engine, max_samples=max_samples, **kwargs)
        elif task == "humaneval":
            from .humaneval import eval_humaneval

            result = eval_humaneval(self.engine, max_samples=max_samples, **kwargs)
        elif task == "throughput":
            from .throughput import eval_throughput

            # throughput ignores max_samples; forward only explicit kwargs.
            result = eval_throughput(self.engine, **kwargs)
        else:
            raise ValueError(f"Unknown benchmark task: {task_name!r}")
        self._last_results[task] = result
        return result

    def run_all(
        self,
        tasks: list[str] | None = None,
        max_samples: int = 50,
    ) -> dict[str, BenchmarkResult]:
        if tasks is None:
            tasks = ["mmlu", "humaneval", "throughput"]
        results: dict[str, BenchmarkResult] = {}
        for t in tasks:
            results[t.lower().strip()] = self.run(t, max_samples=max_samples)
        self._last_results = dict(results)
        return results

    def save_results(self, path: str | Path) -> None:
        """Persist the most recent ``run``/``run_all`` results as JSON."""
        out_path = Path(path)
        payload = {
            name: result.to_dict() if isinstance(result, BenchmarkResult) else result
            for name, result in self._last_results.items()
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    def load_results(self, path: str | Path) -> dict[str, BenchmarkResult]:
        """Load previously saved benchmark results from JSON."""
        in_path = Path(path)
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        loaded: dict[str, BenchmarkResult] = {}
        for name, d in payload.items():
            loaded[name] = BenchmarkResult.from_dict(d)
        self._last_results = dict(loaded)
        return loaded
