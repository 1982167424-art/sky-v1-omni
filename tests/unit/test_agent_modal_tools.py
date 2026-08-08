from sky_v1.agent.base import ToolContext
from sky_v1.agent.tools.image_tools import ImageGenerationTool
from sky_v1.agent.tools.audio_tools import TTSTool
from sky_v1.agent.tools.video_tools import VideoGenerationTool
from sky_v1.agent.tools.three_d_tools import PointCloudTool, MeshTool, NERFTool


def _ctx():
    return ToolContext(session_id="modal_test")


def test_image_generation_tool():
    r = ImageGenerationTool().run(_ctx(), prompt="a cat")
    assert r.success is True
    assert "image_url" in r.data


def test_tts_tool():
    r = TTSTool().run(_ctx(), text="hello world")
    assert r.success is True
    assert "audio_url" in r.data


def test_video_generation_tool():
    r = VideoGenerationTool().run(_ctx(), prompt="walking in park")
    assert r.success is True
    assert "video_url" in r.data


def test_pointcloud_tool():
    r = PointCloudTool().run(_ctx(), prompt="a dog")
    assert r.success is True
    assert "obj_url" in r.data


def test_mesh_tool():
    r = MeshTool().run(_ctx(), prompt="a chair")
    assert r.success is True
    assert "obj_url" in r.data


def test_nerf_tool():
    r = NERFTool().run(_ctx(), prompt="a room")
    assert r.success is True
    assert "ckpt_url" in r.data
