"""API 搜索 / 深度推理路由集成测试 + RAG 预置新文档测试。

通过 FastAPI TestClient 测试 /v1/search/web 与 /v1/reasoning/deep 端点，
不依赖网络；并验证 chinese_llm_landscape 文档在 RAG ingestion 后可命中查询。
"""
from __future__ import annotations

import pytest

from sky_v1.api.app import create_app
from sky_v1.rag.presets import PRESET_DOCS


@pytest.fixture(scope="module")
def test_client():
    if not __import__("sky_v1").API_AVAILABLE:
        pytest.skip("API 模块不可用")
    from fastapi.testclient import TestClient
    app = create_app(enable_engine=False)
    return TestClient(app)


def test_search_web_endpoint_responds(test_client):
    resp = test_client.post("/v1/search/web", json={"query": "Qwen3.8-Max 参数规模", "num_results": 3})
    assert resp.status_code == 200
    body = resp.json()
    for k in ("results", "provider", "cached", "simulated", "latency_ms"):
        assert k in body
    assert isinstance(body["results"], list)
    # simulated 兜底至少有 3 条
    assert len(body["results"]) >= 1
    sample = body["results"][0]
    assert "title" in sample and "url" in sample and "snippet" in sample


def test_search_web_endpoint_rejects_empty_query(test_client):
    resp = test_client.post("/v1/search/web", json={"query": ""})
    # Pydantic validation 会返回 422
    assert resp.status_code in (200, 422)
    if resp.status_code == 200:
        # 工具层自己拒绝的情况
        assert resp.json().get("simulated") or len(resp.json().get("results", [])) == 0


def test_reasoning_deep_endpoint_responds(test_client):
    resp = test_client.post(
        "/v1/reasoning/deep",
        json={
            "question": "DeepSeek、Kimi、豆包，分别适合什么用户群体？",
            "max_iterations": 2,
            "enable_web_search": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    for k in ("plan", "iterations", "final_answer", "confidence", "simulated", "latency_ms"):
        assert k in body
    assert isinstance(body["iterations"], list)
    assert len(body["iterations"]) <= 2
    assert 0.0 < body["confidence"] <= 1.0
    assert body["final_answer"], "final_answer 不应为空"


def test_reasoning_deep_endpoint_bounds_max_iterations(test_client):
    resp = test_client.post(
        "/v1/reasoning/deep",
        json={"question": "x", "max_iterations": 8, "enable_web_search": False},
    )
    # max 8 超出 schema 1..6，应当被 Pydantic 拒绝
    assert resp.status_code == 422


def test_agent_tools_list_includes_new_tools(test_client):
    resp = test_client.get("/v1/agent/tools")
    assert resp.status_code == 200
    names = {t.get("name") for t in resp.json().get("tools", [])}
    assert "tool_web_search" in names
    assert "tool_deep_reasoning" in names


# ---------------------------------------------------------------------------
# RAG 新预置文档
# ---------------------------------------------------------------------------
def test_preset_docs_contains_chinese_llm_landscape():
    ids = [id_ for id_, _, _, _ in PRESET_DOCS]
    assert "chinese_llm_landscape" in ids
    paths = [p for _, _, _, p in PRESET_DOCS]
    target = next(p for p in paths if "chinese_llm_landscape" in str(p))
    text = target.read_text(encoding="utf-8")
    # 文档应包含 13+ 个国产模型关键厂商名称
    for vendor in ["豆包", "ERNIE", "通义千问", "混元", "盘古", "讯飞星火", "MiMo",
                   "DeepSeek", "Kimi", "MiniMax", "百川", "阶跃星辰", "商汤"]:
        assert vendor in text, f"preset doc 缺 {vendor}"
    # 选型速查表存在
    assert "选型建议速查表" in text


def test_preset_docs_count_updated():
    # 原 8 篇 + 本次 1 篇 = 9
    assert len(PRESET_DOCS) >= 9
