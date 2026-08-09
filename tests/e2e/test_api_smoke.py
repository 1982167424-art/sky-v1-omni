VALID_FINISH_REASONS = {"stop", "error", "length", "tool_calls", "content_filter"}


def test_health_ok(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"


def test_metrics_has_requests_total(test_client):
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "requests_total" in body


def test_agent_tools_14(test_client):
    """注册工具数：12 基础 (text/vision/audio/video/3D) + 联网搜索 + 深度推理 = 14"""
    resp = test_client.get("/v1/agent/tools")
    assert resp.status_code == 200
    body = resp.json()
    tools = body.get("tools", [])
    names = {t.get("name") for t in tools}
    assert len(tools) == 14
    assert "tool_web_search" in names
    assert "tool_deep_reasoning" in names


def test_chat_completions(test_client):
    payload = {
        "model": "sky-v1-agent",
        "messages": [{"role": "user", "content": "你好"}],
    }
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    choices = body.get("choices", [])
    assert len(choices) >= 1
    finish_reason = choices[0].get("finish_reason")
    assert finish_reason in VALID_FINISH_REASONS


def test_rag_query_has_results_key(test_client):
    payload = {"query": "Transformer", "top_k": 3}
    resp = test_client.post("/v1/rag/query", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body


def test_image_generations_url(test_client):
    payload = {"prompt": "a cute cat", "n": 1, "size": "1024x1024"}
    resp = test_client.post("/v1/images/generations", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    data = body.get("data", [])
    assert len(data) >= 1
    assert "url" in data[0]


def test_agent_run_has_round(test_client):
    payload = {"user_message": "你好", "session_id": "s1", "user_id": "u1"}
    resp = test_client.post("/v1/agent/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "round" in body
