"""sky_v1.eval: Benchmark evaluation framework (MMLU / HumanEval / throughput).

Implements spec §10.1 Benchmark layer: MMLU/MMBench/HumanEval/inference throughput.
"""
from .benchmark import BenchmarkRunner, BenchmarkResult
from .mmlu import eval_mmlu
from .humaneval import eval_humaneval
from .throughput import eval_throughput

EVAL_AVAILABLE = True

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "eval_mmlu",
    "eval_humaneval",
    "eval_throughput",
    "EVAL_AVAILABLE",
]
