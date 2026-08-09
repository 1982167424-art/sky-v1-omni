"""Unit tests for the sky_v1.eval benchmark framework (spec §10.1)."""
from __future__ import annotations

from sky_v1.inference.engine import SkyInferenceEngine
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig
from sky_v1.eval import (
    BenchmarkRunner,
    BenchmarkResult,
    eval_mmlu,
    eval_humaneval,
    eval_throughput,
)


def _mini_engine() -> SkyInferenceEngine:
    cfg = SkyModelConfig(
        model_name="mini-eval",
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
        max_seq_len=128,
        vocab_size=512,
        image_vocab_size=0,
        audio_vocab_size=0,
        video_vocab_size=0,
        three_d_vocab_size=0,
        modal=ModalConfig(),
        heads=HeadsConfig(),
    )
    return SkyInferenceEngine(cfg, device="cpu", dtype="fp32")


def test_benchmark_result_dataclass():
    """Verify BenchmarkResult serializes / deserializes round-trip."""
    r = BenchmarkResult(
        task_name="mmlu",
        accuracy=0.5,
        total_samples=10,
        correct=5,
        latency_ms_avg=12.3,
        throughput_tokens_per_s=42.0,
        metadata={"foo": "bar"},
    )
    d = r.to_dict()
    assert d["task_name"] == "mmlu"
    assert d["accuracy"] == 0.5
    assert d["correct"] == 5
    assert d["metadata"] == {"foo": "bar"}
    r2 = BenchmarkResult.from_dict(d)
    assert r2 == r


def test_eval_mmlu_mini_runs():
    """Run mini-MMLU on 5 questions; accuracy must be in [0, 1]."""
    engine = _mini_engine()
    result = eval_mmlu(engine, max_samples=5, num_few_shot=2)
    assert isinstance(result, BenchmarkResult)
    assert result.task_name == "mmlu"
    assert result.total_samples == 5
    assert 0.0 <= result.accuracy <= 1.0
    assert 0 <= result.correct <= 5
    assert result.latency_ms_avg >= 0.0


def test_eval_humaneval_mini_runs():
    """Run mini-HumanEval on 2 tasks; returns a BenchmarkResult."""
    engine = _mini_engine()
    result = eval_humaneval(engine, max_samples=2, k=1)
    assert isinstance(result, BenchmarkResult)
    assert result.task_name == "humaneval"
    assert result.total_samples == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert result.metadata.get("metric") == "pass@1"


def test_eval_throughput_runs():
    """Throughput measurement returns positive latency."""
    engine = _mini_engine()
    result = eval_throughput(engine, max_new_tokens=4, warmup=1, repeats=2)
    assert isinstance(result, BenchmarkResult)
    assert result.task_name == "throughput"
    assert result.latency_ms_avg > 0
    assert result.metadata.get("repeats") == 2
    assert "device" in result.metadata
    assert "model_name" in result.metadata
    assert "dtype" in result.metadata


def test_benchmark_runner_save_load_json(tmp_path):
    """Verify JSON round-trip consistency for BenchmarkRunner results."""
    engine = _mini_engine()
    runner = BenchmarkRunner(engine, device="cpu")
    results = runner.run_all(tasks=["throughput"], max_samples=2)
    assert "throughput" in results

    out = tmp_path / "bench_results.json"
    runner.save_results(out)
    assert out.exists()

    # Fresh runner loads and reconstructs BenchmarkResult objects.
    runner2 = BenchmarkRunner(engine, device="cpu")
    loaded = runner2.load_results(out)
    assert "throughput" in loaded
    assert isinstance(loaded["throughput"], BenchmarkResult)
    assert loaded["throughput"].task_name == "throughput"
    # latency survived the round-trip
    assert loaded["throughput"].latency_ms_avg > 0
