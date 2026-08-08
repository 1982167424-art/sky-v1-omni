def test_step_hello_keys(fresh_agent):
    r = fresh_agent.step("你好")
    assert isinstance(r, dict)
    assert "answer" in r
    assert "tool_calls" in r
    assert "plan" in r
    assert len(r["tool_calls"]) >= 1


def test_step_3d_mesh(fresh_agent):
    r = fresh_agent.step("生成3D Mesh")
    assert len(r["tool_calls"]) >= 1
    first = r["tool_calls"][0]
    tool_name = getattr(first, "tool_name", None) or (first.get("tool_name") if isinstance(first, dict) else None)
    assert tool_name == "tool_3d_mesh"


def test_step_memory_writes(fresh_agent):
    fresh_agent.step("你好，第一次消息")
    fresh_agent.step("你好，第二次消息")
    msgs = fresh_agent.memory.get()
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 2
