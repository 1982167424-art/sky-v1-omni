def _tool_name(tc):
    if hasattr(tc, "tool_name"):
        return tc.tool_name
    if isinstance(tc, dict):
        return tc.get("tool_name") or tc.get("tool") or ""
    return ""


def test_asr_for_audio(fresh_agent):
    r = fresh_agent.step("语音识别这个音频文件")
    tcs = r.get("tool_calls", [])
    assert len(tcs) >= 1
    assert _tool_name(tcs[0]) == "tool_asr"


def test_tts_for_text(fresh_agent):
    r = fresh_agent.step("朗读这句话：你好世界")
    tcs = r.get("tool_calls", [])
    assert len(tcs) >= 1
    assert _tool_name(tcs[0]) == "tool_tts"


def test_3d_pointcloud_for_dog(fresh_agent):
    r = fresh_agent.step("3D Point cloud 生成一只狗")
    tcs = r.get("tool_calls", [])
    assert len(tcs) >= 1
    assert _tool_name(tcs[0]) == "tool_3d_pointcloud"
