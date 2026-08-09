"""sky_v1.eval.humaneval: HumanEval-style pass@1 evaluation.

Implements spec §10.1 Benchmark layer. Uses a hard-coded mini-HumanEval of 5
programming tasks. Generated code is executed via ``exec`` inside a try/except
sandbox that refuses dangerous module imports.
"""
from __future__ import annotations

import re
import time
from typing import Any

from .benchmark import BenchmarkResult


MINI_HUMANEXAL_TASKS: list[dict[str, str]] = [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than threshold.\n"
            "    \"\"\"\n"
        ),
        "canonical_solution": (
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        ),
        "test": (
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
            "assert has_close_elements([1.0, 2.8, 3.0], 0.3) == True\n"
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.0], 0.95) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/1",
        "prompt": (
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\" Return the sum of two integers.\n"
            "    \"\"\"\n"
        ),
        "canonical_solution": "    return a + b\n",
        "test": (
            "assert add(1, 2) == 3\n"
            "assert add(-1, 1) == 0\n"
            "assert add(0, 0) == 0\n"
        ),
    },
    {
        "task_id": "HumanEval/2",
        "prompt": (
            "def is_even(n: int) -> bool:\n"
            "    \"\"\" Return True if n is even, False otherwise.\n"
            "    \"\"\"\n"
        ),
        "canonical_solution": "    return n % 2 == 0\n",
        "test": (
            "assert is_even(2) == True\n"
            "assert is_even(3) == False\n"
            "assert is_even(0) == True\n"
        ),
    },
    {
        "task_id": "HumanEval/3",
        "prompt": (
            "def max_of_two(a: int, b: int) -> int:\n"
            "    \"\"\" Return the larger of two integers.\n"
            "    \"\"\"\n"
        ),
        "canonical_solution": "    return a if a > b else b\n",
        "test": (
            "assert max_of_two(1, 2) == 2\n"
            "assert max_of_two(5, 3) == 5\n"
            "assert max_of_two(4, 4) == 4\n"
        ),
    },
    {
        "task_id": "HumanEval/4",
        "prompt": (
            "def multiply(a: int, b: int) -> int:\n"
            "    \"\"\" Return the product of two integers.\n"
            "    \"\"\"\n"
        ),
        "canonical_solution": "    return a * b\n",
        "test": (
            "assert multiply(2, 3) == 6\n"
            "assert multiply(0, 5) == 0\n"
            "assert multiply(-1, 4) == -4\n"
        ),
    },
]


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Sandbox: refuse dangerous module imports and obvious shell/FS escape vectors.
_DANGEROUS_RE = re.compile(
    r"\b(?:import|from)\s+(?:os|sys|subprocess|shutil|socket|http|urllib|requests|"
    r"pickle|ctypes|pathlib|tempfile|signal|multiprocessing|threading|asyncio|"
    r"builtins)\b"
    r"|__import__|os\.system|subprocess\.(?:run|call|Popen)|shutil\.rmtree",
)


def _extract_code(text: str) -> str:
    """Extract a Python code block from model output, falling back to raw text."""
    if not text:
        return ""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    return text


def _is_safe(code: str) -> bool:
    return _DANGEROUS_RE.search(code) is None


def _run_test(code: str, prompt: str, test: str) -> bool:
    """Execute prompt + generated code + test inside a sandboxed namespace.

    Returns True only if the assertions pass; any exception (SyntaxError,
    AssertionError, NameError, ...) or sandbox refusal returns False.
    """
    full = f"{prompt}\n{code}\n{test}"
    if not _is_safe(full):
        return False
    namespace: dict[str, Any] = {"__name__": "__test__"}
    try:
        exec(full, namespace)  # noqa: S102 - intentional sandboxed eval
        return True
    except Exception:
        return False


def eval_humaneval(
    engine: Any,
    max_samples: int = 20,
    k: int = 1,
) -> BenchmarkResult:
    """Run the mini-HumanEval pass@1 evaluation.

    pass@1 = passed / total. For k>1 the metric is still reported as pass@1
    (single-sample); ``k`` is recorded in metadata for future extension.
    """
    tasks = MINI_HUMANEXAL_TASKS[: max(0, min(max_samples, len(MINI_HUMANEXAL_TASKS)))]

    passed = 0
    latencies: list[float] = []
    for t in tasks:
        messages = [{"role": "user", "content": t["prompt"]}]
        t0 = time.perf_counter()
        try:
            out = engine.chat(messages, max_new_tokens=128, temperature=0.0)
            text = out.get("text", "") if isinstance(out, dict) else ""
        except Exception:
            text = ""
        latencies.append((time.perf_counter() - t0) * 1000.0)
        code = _extract_code(text or "")
        if _run_test(code, t["prompt"], t["test"]):
            passed += 1

    total = len(tasks)
    pass_at_1 = (passed / total) if total else 0.0
    latency_avg = (sum(latencies) / len(latencies)) if latencies else 0.0

    return BenchmarkResult(
        task_name="humaneval",
        accuracy=pass_at_1,
        total_samples=total,
        correct=passed,
        latency_ms_avg=latency_avg,
        throughput_tokens_per_s=0.0,
        metadata={"k": k, "metric": "pass@1"},
    )
