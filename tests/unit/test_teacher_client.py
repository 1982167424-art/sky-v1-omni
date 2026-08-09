"""单元测试：5 老师蒸馏 API 客户端 + Qwen2.5-72B 本地 fallback。

测试中不做真实网络请求；所有外部调用均被 try/except 保护。
"""
from __future__ import annotations

import pytest

from sky_v1.training.teacher_client import (
    DEFAULT_TEACHERS,
    DistillSetBuilder,
    Qwen72BLocalTeacher,
    TeacherAPIClient,
)

# 全部 5 个老师的 API key 环境变量名（与 DEFAULT_TEACHERS 一致）
_TEACHER_ENV_KEYS = [
    DEFAULT_TEACHERS[n]["api_key_env"] for n in DEFAULT_TEACHERS
]


def test_teacher_api_client_init():
    """验证 5 老师配置加载：老师名、provider、endpoint、权重齐全。"""
    client = TeacherAPIClient()
    names = client.teacher_names()
    assert names == [
        "claude_opus_4_8",
        "gpt_5_6_sol",
        "kimi_k3",
        "mimo_v2_5",
        "qwen_3_8",
    ]
    assert len(names) == 5
    for n in names:
        cfg = client.teachers[n]
        assert "provider" in cfg and "endpoint" in cfg
        assert "api_key_env" in cfg and "model_id" in cfg
        assert "weight" in cfg
    # 从 yaml 加载也应得到一致的 5 老师
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "training" / "teachers.yaml"
    if cfg_path.exists():
        client_yaml = TeacherAPIClient(config_path=cfg_path)
        assert client_yaml.teacher_names() == names


def test_call_teacher_no_key_returns_unavailable(monkeypatch):
    """无 API key 时调用老师应返回 unavailable，绝不崩溃。"""
    # 确保所有 5 老师的 env key 都被清空
    for k in _TEACHER_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)

    client = TeacherAPIClient()
    for name in client.teacher_names():
        assert client.is_configured(name) is False
        result = client.call_teacher(name, "hello", max_tokens=8)
        assert isinstance(result, dict)
        assert result.get("available") is False
        assert "reason" in result
        # 不应抛异常，且返回结构完整
        assert "text" in result and "usage" in result and "latency_ms" in result

    # 并行调全部老师也应全部 unavailable，不崩溃
    all_res = client.call_all_teachers("hello", max_tokens=8)
    assert isinstance(all_res, dict)
    assert set(all_res.keys()) == set(client.teacher_names())
    for name, r in all_res.items():
        assert r.get("available") is False


def test_qwen72b_fallback_is_available_check():
    """Qwen72BLocalTeacher.is_available() 必须返回 bool 且不抛异常。"""
    qwen = Qwen72BLocalTeacher()
    # 测试环境一般没有 transformers/72B 权重，应返回 False（bool），不应抛异常
    available = qwen.is_available()
    assert isinstance(available, bool)
    # generate 在不可用时也应返回 stub dict，不崩溃
    out = qwen.generate("hello", max_tokens=4)
    assert isinstance(out, dict)
    assert "text" in out and "available" in out
    if not available:
        assert out.get("available") is False
        assert "reason" in out


def test_distillset_builder_from_precomputed_empty(tmp_path):
    """加载空文件 / 不存在文件不应崩溃，返回空 records 的 builder。"""
    empty_pt = tmp_path / "empty.pt"
    empty_pt.write_bytes(b"")  # 真正的空文件

    builder = DistillSetBuilder.from_precomputed(empty_pt)
    assert isinstance(builder, DistillSetBuilder)
    assert builder.records == []

    # 不存在的文件也不崩溃
    missing = tmp_path / "nope.pt"
    builder2 = DistillSetBuilder.from_precomputed(missing)
    assert isinstance(builder2, DistillSetBuilder)
    assert builder2.records == []


def test_teacher_weights_match_spec():
    """验证 5 老师权重严格匹配规格 §6：[1.2, 1.3, 1.4, 1.2, 1.0]。"""
    client = TeacherAPIClient()
    weights = client.teacher_weights()
    assert weights == [1.2, 1.3, 1.4, 1.2, 1.0]
    # 权重顺序必须与 teacher_names() 对齐
    names = client.teacher_names()
    assert len(weights) == len(names) == 5
    # 与 TeacherPool（蒸馏训练侧）的权重保持一致；pool 用 float32 存在精度误差，用 approx 比对
    from sky_v1.training.distill import TeacherPool

    pool = TeacherPool()
    import torch

    pool_w = pool.teacher_weights(torch.device("cpu")).tolist()
    assert pool_w == pytest.approx([1.2, 1.3, 1.4, 1.2, 1.0], rel=1e-5)
    assert pool_w == pytest.approx(weights, rel=1e-5)
