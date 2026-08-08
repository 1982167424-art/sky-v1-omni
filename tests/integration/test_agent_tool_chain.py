def _tool_name(tc):
    if hasattr(tc, "tool_name"):
        return tc.tool_name
    if isinstance(tc, dict):
        return tc.get("tool_name") or tc.get("tool") or ""
    return ""


def test_first_step_calls_some_tool(fresh_agent):
    r = fresh_agent.step("写代码 1+2，然后解释一下")
    tool_calls = r.get("tool_calls", [])
    assert len(tool_calls) >= 1


def test_second_step_generate_image(fresh_agent):
    fresh_agent.step("写代码 1+2，然后解释一下")
    r2 = fresh_agent.step("好的，再生成一张图片")
    tcs = r2.get("tool_calls", [])
    assert len(tcs) >= 1
    first_name = _tool_name(tcs[0])
    assert first_name == "tool_image_generate"
