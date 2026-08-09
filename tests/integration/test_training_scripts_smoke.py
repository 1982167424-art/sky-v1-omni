"""训练脚本冒烟测试：锁定 phase1/2/3 脚本可端到端跑通（含 MetricsLogger.log 调用）。

回归背景：trainer.step() 返回的 metrics 含 `step` 键，phase 脚本曾以
`logger.log(step=step, ..., **metrics)` 重复传 step 导致 TypeError 崩溃。
本测试用 toy 模型配置跑 2 步，确保脚本不再崩溃。
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")  # 训练脚本通过 subprocess 运行，需要 torch

ROOT = Path(__file__).resolve().parents[2]

# 内存安全的 toy 模型配置（mel_bins=128 与 ToyDataGenerator 默认一致）
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


def _toy_config_path() -> str:
    fd, path = tempfile.mkstemp(prefix="sky_toy_model_", suffix=".yaml", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_TOY_YAML)
    return path


def _run(script_module, *args):
    return subprocess.run(
        [sys.executable, "-m", script_module, *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def _run_phase(script: str, tmp_out: str):
    cfg = _toy_config_path()
    try:
        r = _run(f"scripts.training.{script}",
                 "--config", cfg, "--steps", "2", "--device", "cpu",
                 "--output-dir", tmp_out)
    finally:
        os.unlink(cfg)
    return r


def test_phase1_warmup_script_runs_no_crash(tmp_path):
    r = _run_phase("phase1_warmup", str(tmp_path / "p1"))
    assert r.returncode == 0, f"phase1 crashed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "Phase1" in r.stdout


def test_phase2_align_script_runs_no_crash(tmp_path):
    r = _run_phase("phase2_align", str(tmp_path / "p2"))
    assert r.returncode == 0, f"phase2 crashed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "Phase2" in r.stdout


def test_phase3_distill_script_runs_no_crash(tmp_path):
    r = _run_phase("phase3_distill", str(tmp_path / "p3"))
    assert r.returncode == 0, f"phase3 crashed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "Phase3" in r.stdout
