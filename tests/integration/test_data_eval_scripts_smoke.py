"""数据下载 / 评估 / 暖启动脚本冒烟测试。

锁定 M5 交付的 CLI 脚本可端到端跑通：
  * scripts/data/download_pretrain_corpus / download_sft_dataset /
    download_modal_datasets / download_preference —— 多镜像源下载失败时
    回退到 toy 样本（spec §11）。
  * scripts/data/build_distillset —— 5 老师 API + Qwen72B 本地 fallback。
  * scripts/eval/run_benchmark —— MMLU / HumanEval / 吞吐评测。
  * scripts/training/init_from_pretrained —— 预训练权重暖启动（spec §2.2）。

所有脚本均在 ``SKY_V1_TEST_MODE=1`` 下运行，避免真实网络请求；模型类脚本
使用 toy 配置，确保 6GB 内存环境不 OOM。
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 内存安全的 toy 模型配置（与 test_training_scripts_smoke 保持一致）。
_TOY_YAML = """\
model:
  name: sky-v1-toy
  hidden_size: 64
  num_hidden_layers: 1
  num_attention_heads: 4
  intermediate_size: 128
  vocab_size: 200
  max_position_embeddings: 256
  modal_types: ["text", "image", "audio", "video", "three_d"]
  modal:
    text: { vocab_size: 200, modal_id: 0 }
    image: { patch_size: 16, image_size: 32, in_channels: 3, modal_id: 1 }
    audio: { mel_bins: 128, subsample: 4, modal_id: 2 }
    video: { num_frames: 2, patch_size: 16, frame_size: 32, modal_id: 3 }
    three_d: { num_points: 64, point_dim: 6, mesh_vertices: 16, modal_id: 4 }
  heads:
    text: { vocab_size: 200 }
    image: { out_channels: 3, patch_size: 16 }
    audio: { mel_bins: 128 }
    video: { num_frames: 2, out_channels: 3, patch_size: 16 }
    three_d: { num_points: 64, point_dim: 3, mesh_vertices: 16 }
"""


def _env():
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SKY_V1_TEST_MODE": "1"}
    return env


def _run(script_module, *args, timeout=120):
    return subprocess.run(
        [sys.executable, "-m", script_module, *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
        env=_env(),
    )


def _toy_config_path() -> str:
    fd, path = tempfile.mkstemp(prefix="sky_toy_model_", suffix=".yaml", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_TOY_YAML)
    return path


# --------------------------------------------------------------------------- #
# 数据下载脚本（多镜像源 → toy 回退）
# --------------------------------------------------------------------------- #
def test_download_pretrain_corpus_falls_back_to_toy(tmp_path):
    r = _run(
        "scripts.data.download_pretrain_corpus",
        "--output-dir", str(tmp_path / "pretrain"),
        "--source", "hf", "--dataset", "redpajama", "--max-samples", "4",
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    out = tmp_path / "pretrain" / "redpajama.jsonl"
    assert out.exists(), f"输出文件未生成: {out}"
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0]).get("toy") is True
    assert "FALLBACK_TOY" in r.stdout


def test_download_sft_dataset_falls_back_to_toy(tmp_path):
    r = _run(
        "scripts.data.download_sft_dataset",
        "--output-dir", str(tmp_path / "sft"),
        "--source", "aliyun", "--dataset", "alpaca", "--max-samples", "3",
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    out = tmp_path / "sft" / "alpaca_sft.jsonl"
    assert out.exists()
    samples = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().splitlines()]
    assert len(samples) == 3
    # toy SFT 样本应被规范化为 {instruction, input, output}
    assert {"instruction", "input", "output"} <= set(samples[0].keys())
    assert "FALLBACK_TOY" in r.stdout


def test_download_modal_datasets_single_modality(tmp_path):
    r = _run(
        "scripts.data.download_modal_datasets",
        "--output-dir", str(tmp_path / "modal"),
        "--modality", "image", "--dataset", "coco_captions", "--max-samples", "2",
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    out = tmp_path / "modal" / "image_coco_captions.jsonl"
    assert out.exists()
    samples = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().splitlines()]
    assert len(samples) == 2
    assert samples[0].get("modality") == "image"
    assert samples[0].get("toy") is True


def test_download_preference_falls_back_to_toy(tmp_path):
    r = _run(
        "scripts.data.download_preference",
        "--output-dir", str(tmp_path / "pref"),
        "--source", "modelscope", "--dataset", "ultrafeedback", "--max-samples", "2",
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    out = tmp_path / "pref" / "ultrafeedback_preference.jsonl"
    assert out.exists()
    samples = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().splitlines()]
    assert len(samples) == 2
    assert {"prompt", "chosen", "rejected"} <= set(samples[0].keys())
    assert "FALLBACK_TOY" in r.stdout


# --------------------------------------------------------------------------- #
# 5 老师蒸馏数据集构建（无 API key → 全 unavailable；--no-fallback 避免加载 Qwen72B）
# --------------------------------------------------------------------------- #
def test_build_distillset_runs_without_api_keys(tmp_path):
    r = _run(
        "scripts.data.build_distillset",
        "--output", str(tmp_path / "distill.pt"),
        "--max-samples", "2", "--no-fallback",
        timeout=180,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    assert "BuildDistillSet" in r.stdout
    assert (tmp_path / "distill.pt").exists()


# --------------------------------------------------------------------------- #
# Benchmark 评估（小样本，CPU）
# --------------------------------------------------------------------------- #
def test_run_benchmark_smoke(tmp_path):
    out = tmp_path / "results.json"
    r = _run(
        "scripts.eval.run_benchmark",
        "--tasks", "mmlu,humaneval,throughput",
        "--max-samples", "3", "--device", "cpu",
        "--output", str(out), "--model", "mini-bench",
        timeout=180,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    # 结果表头与至少一个任务行
    assert "Task" in r.stdout
    assert "mmlu" in r.stdout
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "mmlu" in data


# --------------------------------------------------------------------------- #
# 预训练权重暖启动（toy 配置，未提供 repo → 全部 skipped）
# --------------------------------------------------------------------------- #
def test_init_from_pretrained_smoke(tmp_path):
    cfg = _toy_config_path()
    out = tmp_path / "warm.ckpt"
    try:
        r = _run(
            "scripts.training.init_from_pretrained",
            "--model-config", cfg, "--output", str(out),
            timeout=120,
        )
    finally:
        os.unlink(cfg)
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    assert "init_from_pretrained" in r.stdout
    assert out.exists()
    # 未传 --text-repo/--image-repo/--audio-repo，三模态应全部 skipped
    assert "SKIP" in r.stdout
