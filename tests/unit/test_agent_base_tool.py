import pytest

from sky_v1.agent.base import ToolResult


EXPECTED_TOOL_NAMES = {
    "tool_chat",
    "tool_code",
    "tool_rag",
    "tool_image_understand",
    "tool_image_generate",
    "tool_asr",
    "tool_tts",
    "tool_video_understand",
    "tool_video_generate",
    "tool_3d_pointcloud",
    "tool_3d_mesh",
    "tool_3d_nerf",
}


def test_all_12_tools_registered(fresh_registry):
    names = set(fresh_registry.names())
    assert len(names) == 12
    assert names == EXPECTED_TOOL_NAMES
    assert "tool_chat" in names


def test_register_type_wrong(fresh_registry):
    with pytest.raises(TypeError):
        fresh_registry.register("not a tool")


def test_tool_result_pydantic():
    tr = ToolResult(success=True, output="ok")
    dumped = tr.model_dump()
    assert "success" in dumped
    assert "output" in dumped
    assert "error" in dumped
