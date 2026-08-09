"""CLI entrypoint for the sky-v1 Benchmark evaluation framework.

Builds a direct-engine SkySDK, dispatches the requested benchmark tasks via
``BenchmarkRunner.run_all``, prints a result table, and persists results to
JSON.

Usage:
  python -m scripts.eval.run_benchmark --tasks mmlu,humaneval,throughput
  python -m scripts.eval.run_benchmark --tasks throughput --max-samples 20 \
      --output results.json --model mini-bench
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("run_benchmark")
    p.add_argument(
        "--tasks",
        default="mmlu,humaneval,throughput",
        help="Comma-separated task names (mmlu, humaneval, throughput).",
    )
    p.add_argument("--max-samples", type=int, default=50, help="Max samples per task.")
    p.add_argument("--output", default="benchmark_results.json", help="JSON output path.")
    p.add_argument("--model", default="mini-bench", help="Model name for the SDK.")
    p.add_argument("--device", default="cpu", help="Inference device (cpu/cuda).")
    return p.parse_args()


def _print_table(results: dict) -> None:
    header = (
        f"{'Task':<12} {'Accuracy':>10} {'Samples':>8} {'Correct':>8} "
        f"{'Latency(ms)':>12} {'Throughput(tok/s)':>18}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<12} {r.accuracy:>10.4f} {r.total_samples:>8} {r.correct:>8} "
            f"{r.latency_ms_avg:>12.2f} {r.throughput_tokens_per_s:>18.2f}"
        )


def main() -> int:
    args = parse_args()
    from sky_v1.sdk.client import SkySDK
    from sky_v1.eval import BenchmarkRunner

    sdk = SkySDK(engine="direct", model_name=args.model, device=args.device)
    runner = BenchmarkRunner(sdk, device=args.device)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = runner.run_all(tasks=tasks, max_samples=args.max_samples)

    _print_table(results)
    out_path = Path(args.output)
    runner.save_results(out_path)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
