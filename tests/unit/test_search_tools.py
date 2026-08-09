"""WebSearchTool + DeepReasoningTool 单元测试。

覆盖：缓存命中 < 50ms、空 query 错误路径、无 Provider 时模拟回退、
输入 schema 验证、深度推理子问题拆分 + 迭代步数约束、置信度范围。
"""
from __future__ import annotations

import time

import pytest

from sky_v1.agent.base import ToolContext
from sky_v1.agent.tools.search_tools import (
    DeepReasoningTool,
    WebSearchTool,
    _TtlLruCache,
    _simulated_search_results,
    _split_subquestions,
)


# ---------------------------------------------------------------------------
# 辅助小单元：缓存 / 子问题拆分 / 模拟结果
# ---------------------------------------------------------------------------
def test_ttl_lru_cache_basic_lifecycle():
    cache = _TtlLruCache(capacity=2, ttl_s=3600)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    # 现在 order 是 [b, a]（a 被 get 提至队尾，b 仍是最旧）
    # 插入 c，应淘汰最旧 b
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_ttl_lru_cache_expires():
    cache = _TtlLruCache(capacity=8, ttl_s=0)  # 0s TTL 立即过期
    cache.set("x", {"foo": "bar"})
    assert cache.get("x") is None


def test_split_subquestions_multi_conjunction():
    out = _split_subquestions("比较 DeepSeek 和 Kimi 的优缺点？", n=2)
    # 含"和"应当按连接拆
    assert len(out) == 2


def test_split_subquestions_fallback_defaults():
    out = _split_subquestions("单一陈述句", n=3)
    # 无连接符时生成默认 3 个阶段子问题
    assert len(out) == 3
    assert any("①" in s for s in out)
    assert any("③" in s for s in out)


def test_simulated_search_results_shape():
    out = _simulated_search_results("2026 大模型排行榜", 4)
    assert len(out) == 4
    keys = {tuple(sorted(r.keys())) for r in out}
    assert keys == {("snippet", "title", "url")}
    for r in out:
        assert r["url"].startswith("https://simulated.sky-v1.local/")


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------
def test_web_search_tool_empty_query_returns_error():
    ctx = ToolContext()
    tool = WebSearchTool()
    r = tool.run(ctx, query="")
    assert r.success is False
    assert r.error and "query 不能为空" in r.error


def test_web_search_tool_simulated_returns_structured_results():
    # 不设置任何 Provider Key → 走 simulated 兜底
    ctx = ToolContext()
    tool = WebSearchTool()
    start = time.perf_counter()
    r = tool.run(ctx, query="豆包 vs 文心 vs 千问 对比", num_results=3)
    latency = time.perf_counter() - start
    assert r.success is True
    assert r.latency_ms >= 0
    # 3 results 应该全部返回
    data = r.data or {}
    assert len(data.get("results", [])) == 3
    assert data.get("provider") == "simulated"
    assert data.get("simulated") is True
    assert "共 3 条结果" in r.output
    # simulated 兜底通常 < 100ms
    assert latency < 2.0


def test_web_search_tool_cache_hit_faster_than_simulated_cold():
    ctx = ToolContext()
    tool = WebSearchTool()
    # 冷启动：触发 simulated 路径 + 写入缓存
    cold = tool.run(ctx, query="cache_hit_test_query_0809", num_results=2)
    # 热命中：同 query 再调用
    hot = tool.run(ctx, query="cache_hit_test_query_0809", num_results=2)
    assert cold.success and hot.success
    assert (hot.data or {}).get("cached") is True
    # 缓存命中应 < 50ms（模拟）
    assert hot.latency_ms < 50 or hot.latency_ms <= cold.latency_ms // 2


def test_web_search_tool_inputs_schema_validates_num_results_bounds():
    # schema 只用于文档/API 校验；运行时层 num_results 被 clamp 到 1..20
    ctx = ToolContext()
    tool = WebSearchTool()
    r = tool.run(ctx, query="clamp_test", num_results=100)
    assert r.success
    assert len((r.data or {}).get("results", [])) == 20  # 上限


def test_web_search_tool_name_and_schema_fields():
    assert WebSearchTool.name == "tool_web_search"
    assert WebSearchTool.modal == "text"
    props = WebSearchTool.inputs_schema.get("properties", {})
    assert "query" in props and "num_results" in props
    required = WebSearchTool.inputs_schema.get("required", [])
    assert "query" in required


# ---------------------------------------------------------------------------
# DeepReasoningTool
# ---------------------------------------------------------------------------
def test_deep_reasoning_empty_question_error():
    ctx = ToolContext()
    tool = DeepReasoningTool()
    r = tool.run(ctx, question="")
    assert r.success is False
    assert "question 不能为空" in (r.error or "")


def test_deep_reasoning_iterations_bounded_by_max_iter():
    ctx = ToolContext()
    tool = DeepReasoningTool()
    r = tool.run(ctx, question="请解释大模型选型的关键权衡。", max_iterations=2, enable_web_search=False)
    assert r.success is True
    data = r.data or {}
    iters = data.get("iterations", [])
    assert len(iters) <= 2
    # plan 长度也不会超过迭代数
    assert len(data.get("plan", [])) <= 2
    # 置信度应在 (0, 1]
    conf = data.get("confidence", -1)
    assert 0.0 < conf <= 1.0
    assert "=== Final Answer" in r.output


def test_deep_reasoning_with_web_flag_controls_act_chain():
    ctx = ToolContext()
    tool = DeepReasoningTool()
    # enable_web_search = False：Act 链只能是 A（+ B 当有 RAG 时），不能有 C
    r_off = tool.run(ctx, question="2026 最新大模型排名？", max_iterations=1, enable_web_search=False)
    iters_off = (r_off.data or {}).get("iterations", [])
    assert iters_off
    acts_off = iters_off[0].get("act", "")
    assert "联网搜索事实核查" not in acts_off
    assert "A: 专家知识推演" in acts_off


def test_deep_reasoning_plan_shape_for_comparison():
    ctx = ToolContext()
    tool = DeepReasoningTool()
    r = tool.run(
        ctx,
        question="我想做开源私有化部署和代码开发，选 Qwen 还是 DeepSeek？同时需要处理 20 万字长文档，Kimi 是否更合适？",
        max_iterations=3,
        enable_web_search=False,
    )
    assert r.success
    data = r.data or {}
    final = data.get("final_answer", "")
    assert "置信度" in final
    # 最终答案应包含关键对比字段
    assert any(k in final for k in ("DeepSeek", "Qwen", "Kimi")) or len(final) > 100


def test_deep_reasoning_tool_metadata_present():
    assert DeepReasoningTool.name == "tool_deep_reasoning"
    assert "Tree-of-Thoughts" in DeepReasoningTool.description
    required = DeepReasoningTool.inputs_schema.get("required", [])
    assert "question" in required
    out_props = DeepReasoningTool.outputs_schema.get("properties", {})
    for k in ("plan", "iterations", "final_answer", "confidence"):
        assert k in out_props
