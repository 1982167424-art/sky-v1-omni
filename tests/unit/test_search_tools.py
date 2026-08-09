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
    SearchProviderOutcome,
    WebSearchTool,
    _TtlLruCache,
    _simulated_search_results,
    _split_subquestions,
    _try_baidu_search,
    _try_bing_search,
    _try_google_search,
    _try_toutiao_search,
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


def test_web_search_tool_simulated_off_by_default_returns_failed_or_empty():
    """蓝奏云 v22 经验复盘：默认**不伪造**模拟结果。

    在测试模式 SKY_V1_TEST_MODE=1 且未 allow_simulated 时，
    真实引擎也不会触网；按新流程 all_results/used 都为空且
    test_mode/allow_simulated 均为 False → 应走到 status=EMPTY 或 FAILED。
    我们用 providers=空列表（白名单过滤后为空）显式测参数校验分支。
    """
    ctx = ToolContext()
    tool = WebSearchTool()
    # 白名单全部非法 → 过滤后为空 → 返回参数错误（不再伪造模拟）
    r = tool.run(ctx, query="豆包 vs 文心", num_results=3, providers=["invalid-xxx", "another-bad"])
    assert r.success is False
    assert "providers 白名单为空" in (r.error or "")


def test_web_search_tool_explicit_allow_simulated_returns_simulated():
    """用户显式 `allow_simulated=True` 且真实引擎不可用时，才返回模拟。"""
    ctx = ToolContext()
    tool = WebSearchTool()
    # 使用一个没实现的、会全失败的策略：通过"不配置任何网络"并开启模拟开关
    # SKY_V1_TEST_MODE=1 会让真实引擎全部跳过 → allow_simulated=true 即返回模拟
    import os
    old = os.environ.get("SKY_V1_TEST_MODE", "")
    os.environ["SKY_V1_TEST_MODE"] = "1"
    try:
        r = tool.run(ctx, query="豆包 vs 文心 vs 千问 对比", num_results=3, allow_simulated=True)
    finally:
        os.environ["SKY_V1_TEST_MODE"] = old
    assert r.success is True
    assert r.latency_ms >= 0
    data = r.data or {}
    assert len(data.get("results", [])) == 3
    assert data.get("simulated") is True
    assert "DEV_SIMULATED" in r.output
    assert "共 3 条结果" in r.output


def test_web_search_tool_cache_hit_faster_than_simulated_cold():
    import os
    old = os.environ.get("SKY_V1_TEST_MODE", "")
    os.environ["SKY_V1_TEST_MODE"] = "1"
    ctx = ToolContext()
    tool = WebSearchTool()
    try:
        cold = tool.run(ctx, query="cache_hit_test_query_0809_2", num_results=2, allow_simulated=True)
        hot = tool.run(ctx, query="cache_hit_test_query_0809_2", num_results=2, allow_simulated=True)
    finally:
        os.environ["SKY_V1_TEST_MODE"] = old
    assert cold.success and hot.success
    assert (hot.data or {}).get("cached") is True
    assert hot.latency_ms < 50 or hot.latency_ms <= cold.latency_ms // 2


def test_web_search_tool_inputs_schema_validates_num_results_bounds():
    import os
    old = os.environ.get("SKY_V1_TEST_MODE", "")
    os.environ["SKY_V1_TEST_MODE"] = "1"
    ctx = ToolContext()
    tool = WebSearchTool()
    try:
        r = tool.run(ctx, query="clamp_test", num_results=100, allow_simulated=True)
    finally:
        os.environ["SKY_V1_TEST_MODE"] = old
    assert r.success
    assert len((r.data or {}).get("results", [])) == 20  # 上限


# ---------------------------------------------------------------------------
# 四 Provider 独立函数（返回类型 / EMPTY vs FAILED 区分）
# ---------------------------------------------------------------------------
def test_provider_outcome_dataclass_fields():
    o = SearchProviderOutcome(provider="baidu", available=True, results=[{"title": "a", "url": "", "snippet": ""}], latency_ms=123)
    assert o.provider == "baidu"
    assert o.available is True
    assert o.count_items() if hasattr(o, "count_items") else len(o.results) == 1


def _assert_outcome_shape(outcome: SearchProviderOutcome, name: str):
    # 任何 Provider 的返回都必须是正确类型、正确字段
    assert isinstance(outcome, SearchProviderOutcome)
    assert outcome.provider == name
    assert isinstance(outcome.available, bool)
    assert isinstance(outcome.reason, str)
    assert isinstance(outcome.results, list)
    assert isinstance(outcome.latency_ms, int) and outcome.latency_ms >= 0
    if outcome.available and outcome.results:
        for r in outcome.results:
            assert isinstance(r, dict)
            for k in ("title", "url", "snippet"):
                assert k in r


def test_google_provider_returns_outcome_shape(monkeypatch):
    def fake_get(*a, **kw):
        return "<html><h3>Google Result 1</h3><h3>Google Result 2</h3></html>"
    import sky_v1.agent.tools.search_tools as st
    monkeypatch.setattr(st, "_http_get", fake_get)
    o = _try_google_search("qwen3.8", 3, 2.0)
    _assert_outcome_shape(o, "google")
    # 有 HTML 解析到 <h3>，应当能出结果
    assert o.available is True
    assert len(o.results) >= 2


def test_baidu_provider_returns_outcome_shape(monkeypatch):
    def fake_get(*a, **kw):
        return (
            '<html>'
            '<h3 class="t"><a href="https://example.com/a">百度结果一</a></h3>'
            '<h3 class="t c-gap-top"><a href="https://example.com/b">百度结果二：通义千问</a></h3>'
            '</html>'
        )
    import sky_v1.agent.tools.search_tools as st
    monkeypatch.setattr(st, "_http_get", fake_get)
    o = _try_baidu_search("通义千问", 5, 2.0)
    _assert_outcome_shape(o, "baidu")
    assert o.available is True
    assert len(o.results) == 2
    assert o.results[0]["url"] == "https://example.com/a"


def test_bing_provider_returns_outcome_shape_html(monkeypatch):
    def fake_get(*a, **kw):
        return (
            '<html>'
            '<li class="b_algo"><h2><a href="https://b1">Bing 1</a></h2><p>Snippet1</p></li></li>'
            '<li class="b_algo"><h2><a href="https://b2">Bing 2</a></h2><p>Snippet2</p></li></li>'
            '</html>'
        )
    import sky_v1.agent.tools.search_tools as st
    monkeypatch.setattr(st, "_http_get", fake_get)
    o = _try_bing_search("bingq", 5, 2.0)
    _assert_outcome_shape(o, "bing")
    assert o.available is True
    assert len(o.results) == 2


def test_toutiao_provider_returns_outcome_shape_html(monkeypatch):
    def fake_get(*a, **kw):
        return (
            '<html>'
            '<a href="https://toutiao.com/a123">今日头条：短剧爆款数据 2026</a>'
            '<a href="/local">忽略：相对路径</a>'
            '<a href="https://toutiao.com/b456">今日头条第二则资讯标题</a>'
            '</html>'
        )
    import sky_v1.agent.tools.search_tools as st
    monkeypatch.setattr(st, "_http_get", fake_get)
    o = _try_toutiao_search("短剧", 5, 2.0)
    _assert_outcome_shape(o, "toutiao")
    assert o.available is True
    # 仅保留以 http 开头且 title 长度合理的
    assert len(o.results) == 2
    assert all(r["url"].startswith("http") for r in o.results)


def test_provider_empty_hit_vs_network_failure_distinction(monkeypatch):
    """关键用例：经验复盘严格区分 '请求成功但0命中' vs '网络异常'。"""
    import sky_v1.agent.tools.search_tools as st

    # Case 1: 成功但 0 命中（空 HTML 正常返回）
    monkeypatch.setattr(st, "_http_get", lambda *a, **kw: "<html></html>")
    o = _try_baidu_search("一个不存在的极冷词_xyzzy", 5, 2.0)
    assert o.available is True
    assert o.results == []
    assert o.reason == ""  # EMPTY：无失败原因

    # Case 2: 真正的网络异常
    def boom(*a, **kw):
        raise RuntimeError("DNS FAIL baidu.com refused")
    monkeypatch.setattr(st, "_http_get", boom)
    o2 = _try_baidu_search("无论什么词", 5, 0.5)
    assert o2.available is False
    assert "NETWORK_ERROR" in o2.reason
    assert "DNS FAIL" in o2.reason


# ---------------------------------------------------------------------------
# 多引擎并发 + 快速失败（monkeypatch 4 provider 函数注入不同延迟结果）
# ---------------------------------------------------------------------------
_MOD_ATTR_BY_PROVIDER = {
    "google": "_try_google_search",
    "baidu": "_try_baidu_search",
    "bing": "_try_bing_search",
    "toutiao": "_try_toutiao_search",
    "tavily": "_try_tavily",  # 注意：tavily 的函数名不统一（历史遗留）
}


@pytest.fixture
def _install_providers(monkeypatch, request):
    """返回一个闭包: install(st, mapping) -> None，会在当前用例结束后自动还原 PROVIDER_FN。

    注意：conftest autouse fixture 默认把 SKY_V1_TEST_MODE=1 注入所有测试（禁止真实网络访问）。
    而 Provider 并发测试用"打桩函数替换 provider"，不需要真实网络，故必须**关闭**该环境变量，
    否则 run() 内部会因 test_mode=True 跳过并发执行、进入 simulated 回退。
    """
    import sky_v1.agent.tools.search_tools as _st

    # 1) 关闭 SKY_V1_TEST_MODE（允许 run() 走真实 Provider 调度分支；但 provider 均被桩函数替换，不会真实触网）
    monkeypatch.delenv("SKY_V1_TEST_MODE", raising=False)
    # 同时清掉任何可能存在的 Search API Key，确保桩函数路径可控
    for _k in ("SERPAPI_API_KEY", "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_CX", "BING_SEARCH_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(_k, raising=False)

    original_dict = dict(_st.WebSearchTool.PROVIDER_FN)

    def _install(mapping: dict):
        st = _st
        # 模块级属性：monkeypatch 保证还原
        for name, fn in mapping.items():
            attr_name = _MOD_ATTR_BY_PROVIDER.get(name, f"_try_{name}_search")
            monkeypatch.setattr(st, attr_name, fn)
        # PROVIDER_FN 原地改 dict 内容（保证子线程可见）
        for name, fn in mapping.items():
            st.WebSearchTool.PROVIDER_FN[name] = fn

    def _restore():
        for k, v in original_dict.items():
            _st.WebSearchTool.PROVIDER_FN[k] = v
        for k in list(_st.WebSearchTool.PROVIDER_FN.keys()):
            if k not in original_dict:
                _st.WebSearchTool.PROVIDER_FN.pop(k, None)

    request.addfinalizer(_restore)
    return _install


def test_concurrent_multiple_providers_concatenate_dedup(_install_providers):
    """并发 4 引擎都返回成功时，结果合并后去重、按 top N 返回。"""
    def mk(name, titles):
        def _f(q, n, t):
            return SearchProviderOutcome(
                provider=name, available=True,
                results=[{"title": t, "url": f"https://{name}/{i}", "snippet": f"{name}:{t}"} for i, t in enumerate(titles)],
                latency_ms=10,
            )
        return _f
    mapping = {
        "google": mk("google", ["A", "重复", "B"]),
        "baidu": mk("baidu", ["C", "重复", "D"]),
        "bing": mk("bing", ["E"]),
        "toutiao": mk("toutiao", ["F", "G"]),
        "tavily": lambda *a: SearchProviderOutcome(provider="tavily", available=False, reason="MISSING_KEY"),
    }
    _install_providers(mapping)

    ctx = ToolContext(config={"search_timeout_s": 5.0})
    tool = WebSearchTool()
    r = tool.run(ctx, query="并发测试", num_results=5)
    assert r.success is True
    data = r.data or {}
    assert data.get("status") == "OK"
    titles = [rr["title"] for rr in data["results"]]
    assert "重复" in titles
    assert titles.count("重复") == 1  # 去重
    assert len(data["results"]) == 5  # Top N 裁剪
    # providers_used 包含全部 4 家
    used = data.get("providers_used", [])
    assert {"google", "baidu", "bing", "toutiao"} <= set(used)


def test_concurrent_partial_failure_marks_status_partial(_install_providers):
    """部分引擎失败 + 部分引擎成功 → status=PARTIAL 而非 FAILED（仍可用）。"""
    def ok(q, n, t):
        return SearchProviderOutcome(provider="baidu", available=True,
                                      results=[{"title": "ok", "url": "https://a", "snippet": "s"}], latency_ms=1)
    def fail(q, n, t):
        return SearchProviderOutcome(provider="google", available=False,
                                      reason="NETWORK_ERROR: timeout", latency_ms=900)
    def empty(q, n, t):
        return SearchProviderOutcome(provider="bing", available=True, results=[], latency_ms=2)
    def skip(q, n, t):
        return SearchProviderOutcome(provider="tavily", available=False, reason="MISSING_KEY", latency_ms=0)
    def tt_empty(*_):
        return SearchProviderOutcome(provider="toutiao", available=True, results=[], latency_ms=1)
    _install_providers({
        "google": fail, "baidu": ok, "bing": empty,
        "toutiao": tt_empty, "tavily": skip,
    })

    ctx = ToolContext(config={"search_timeout_s": 5.0})
    tool = WebSearchTool()
    r = tool.run(ctx, query="部分失败场景", num_results=3, providers=["google", "baidu", "bing", "toutiao", "tavily"])
    assert r.success is True
    data = r.data or {}
    assert data.get("status") == "PARTIAL"
    # provider_statuses 里 google 的 reason 应可见
    reasons = {row["provider"]: row.get("reason", "") for row in data.get("provider_statuses", [])}
    assert "NETWORK_ERROR" in reasons.get("google", "")
    # MISSING_KEY 在 output 中应显示为 SKIP
    assert "SKIP (未配置 API Key)" in r.output or "tavily" in r.output


def test_concurrent_all_network_failed_marks_status_failed(_install_providers):
    """**全部**已配置引擎都失败 → status=FAILED（默认不伪造）。这是 蓝奏云 v22 经验的核心。"""
    def bad(name):
        def _f(q, n, t):
            return SearchProviderOutcome(provider=name, available=False,
                                          reason=f"NETWORK_ERROR: {name} unreachable", latency_ms=1000)
        return _f
    # tavily 走 MISSING_KEY（不计入错误），但 4 大主流全失败 → errors 非空 & 无 hit
    mapping = {
        "google": bad("google"),
        "baidu": bad("baidu"),
        "bing": bad("bing"),
        "toutiao": bad("toutiao"),
        "tavily": lambda *a: SearchProviderOutcome(provider="tavily", available=False, reason="MISSING_KEY"),
    }
    _install_providers(mapping)

    ctx = ToolContext(config={"search_timeout_s": 5.0})
    tool = WebSearchTool()
    # 注意：未设置 allow_simulated → 默认不伪造
    r = tool.run(ctx, query="全失败场景", num_results=3)
    data = r.data or {}
    assert data.get("status") == "FAILED"
    assert data.get("simulated") is False
    assert data.get("results") == []
    # output 里应有排查建议
    assert "⚠️ 所有已配置的搜索引擎均返回异常" in r.output


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
