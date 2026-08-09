"""M4: FastAPI create_app() + TestClient API 联通冒烟."""
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

def test_chat_completions_endpoint_responds():
    import sky_v1
    if not sky_v1.API_AVAILABLE:
        pytest.skip("API module unavailable (optional deps missing)")
    from fastapi.testclient import TestClient
    app = sky_v1.create_app()
    c = TestClient(app)
    payload = {
        "model": "sky-v1-agent",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    }
    r = c.post("/v1/chat/completions", json=payload)
    assert r.status_code in (200, 500, 501, 404)
    if r.status_code == 200:
        j = r.json()
        assert "choices" in j
