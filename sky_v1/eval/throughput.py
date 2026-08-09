"""sky_v1.eval.throughput: inference throughput evaluation.

Implements spec §10.1 Benchmark layer. Measures average / peak token throughput
and time-to-first-token (TTFT) proxy via repeated single-shot calls.
"""
from __future__ import annotations

import time
from typing import Any

from .benchmark import BenchmarkResult


def eval_throughput(
    engine: Any,
    prompt: str = "The sky is blue",
    max_new_tokens: int = 32,
    warmup: int = 2,
    repeats: int = 5,
) -> BenchmarkResult:
    """Measure inference throughput.

    ``warmup`` calls are timed but discarded; the following ``repeats`` calls
    are averaged. Throughput is approximated as generated_tokens / elapsed_s.
    TTFT is approximated as the first measured (post-warmup) call latency since
    the engine interface is non-streaming.
    """
    # Warmup (not recorded).
    for _ in range(max(0, warmup)):
        try:
            engine.chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        except Exception:
            pass

    latencies_ms: list[float] = []
    token_counts: list[int] = []
    for _ in range(max(0, repeats)):
        t0 = time.perf_counter()
        try:
            out = engine.chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            n_tokens = len(out.get("token_ids", []) or []) if isinstance(out, dict) else 0
        except Exception:
            n_tokens = 0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        token_counts.append(max(n_tokens, 1))

    n = len(latencies_ms)
    latency_avg = (sum(latencies_ms) / n) if n else 0.0
    avg_tokens = (sum(token_counts) / n) if n else 1.0
    avg_throughput = (
        avg_tokens / (latency_avg / 1000.0) if latency_avg > 0 else 0.0
    )
    peak_throughput = 0.0
    for n_tok, lat in zip(token_counts, latencies_ms):
        if lat > 0:
            peak_throughput = max(peak_throughput, n_tok / (lat / 1000.0))
    ttft_ms = latencies_ms[0] if latencies_ms else 0.0

    # Collect device / model metadata defensively (engine may be a stub).
    raw_device = getattr(engine, "device", None)
    raw_dtype = getattr(engine, "dtype", None)
    cfg = getattr(engine, "config", None)
    model_name = getattr(cfg, "name", None) or getattr(cfg, "model_name", None) or "unknown"

    return BenchmarkResult(
        task_name="throughput",
        accuracy=0.0,
        total_samples=n,
        correct=0,
        latency_ms_avg=latency_avg,
        throughput_tokens_per_s=avg_throughput,
        metadata={
            "device": str(raw_device) if raw_device is not None else "cpu",
            "model_name": model_name,
            "dtype": str(raw_dtype) if raw_dtype is not None else "fp32",
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "repeats": repeats,
            "ttft_ms": ttft_ms,
            "peak_throughput_tokens_per_s": peak_throughput,
            "avg_tokens_per_call": avg_tokens,
        },
    )
