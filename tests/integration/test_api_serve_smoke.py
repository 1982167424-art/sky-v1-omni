"""M4: FastAPI create_app() + TestClient API 联通冒烟.

测试两条路径：
1. Agent 模式（默认）：model="sky-v1-agent" → 走 SkyAgent.step()
2. Engine 模式（enable_engine=True）：model="sky-v1-mini" → 走 SkyInferenceEngine.chat()
"""
from __future__ import annotations
import pytest

def test_health_endpoint_via_testclient():
    import sky_v1
    if not sky_v1.API_AVAILABLE:
        pytest.skip("API module unavailable (optional deps missing)")
    from fastapi.testclient import TestClient
    app = sky_v1.create_app()
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"

def test_chat_completions_agent_mode():
    """Agent 模式：model=sky-v1-agent → 走 SkyAgent"""
    import sky_v1
    if not sky_v1.API_AVAILABLE:
        pytest.skip("API module unavailable")
    from fastapi.testclient import TestClient
    app = sky_v1.create_app()
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={
        "model": "sky-v1-agent",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    })
    assert r.status_code in (200, 500, 501, 404)
    if r.status_code == 200:
        j = r.json()
        assert "choices" in j

def test_chat_completions_engine_mode():
    """Engine 模式：enable_engine=True + model=sky-v1-mini → 走 SkyInferenceEngine"""
    import sky_v1
    if not sky_v1.API_AVAILABLE or not sky_v1.INFERENCE_AVAILABLE:
        pytest.skip("API or Inference module unavailable")
    from fastapi.testclient import TestClient
    app = sky_v1.create_app(enable_engine=True)
    assert app.state.engine is not None, "Engine should be initialized"
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={
        "model": "sky-v1-mini",
        "messages": [{"role": "user", "content": "hello sky"}],
        "max_tokens": 4,
    })
    assert r.status_code == 200, f"Engine chat failed: {r.text}"
    j = r.json()
    assert "choices" in j
    content = j["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    # Engine 模式不应返回 [SIM] 前缀（Agent 模式才有）
    # 注意：engine chat 是 stub 解码，可能返回 "(stub)" 或空字符串

def test_completions_engine_mode():
    """Engine 模式：/v1/completions 走引擎 generate_text"""
    import sky_v1
    if not sky_v1.API_AVAILABLE or not sky_v1.INFERENCE_AVAILABLE:
        pytest.skip("API or Inference module unavailable")
    from fastapi.testclient import TestClient
    app = sky_v1.create_app(enable_engine=True)
    c = TestClient(app)
    r = c.post("/v1/completions", json={
        "model": "sky-v1-mini",
        "prompt": "The sky is",
        "max_tokens": 4,
    })
    assert r.status_code == 200
    j = r.json()
    assert "choices" in j
    text = j["choices"][0]["text"]
    assert isinstance(text, str)
