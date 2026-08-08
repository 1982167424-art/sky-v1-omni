"""sky_v1.agent.tools: 11 built-in tools + pre-populated TOOL_REGISTRY."""
from sky_v1.agent.base import ToolRegistry
from sky_v1.agent.tools.audio_tools import ASRTool, TTSTool
from sky_v1.agent.tools.image_tools import ImageGenerationTool, ImageUnderstandingTool
from sky_v1.agent.tools.text_tools import ChatTool, CodeTool, RagTool
from sky_v1.agent.tools.three_d_tools import MeshTool, NERFTool, PointCloudTool
from sky_v1.agent.tools.video_tools import VideoGenerationTool, VideoUnderstandingTool

ChatTool.__qualname__
CodeTool.__qualname__
RagTool.__qualname__
ImageUnderstandingTool.__qualname__
ImageGenerationTool.__qualname__
ASRTool.__qualname__
TTSTool.__qualname__
VideoUnderstandingTool.__qualname__
VideoGenerationTool.__qualname__
PointCloudTool.__qualname__
MeshTool.__qualname__
NERFTool.__qualname__

default_registry = ToolRegistry()
default_registry.register(ChatTool())
default_registry.register(CodeTool())
default_registry.register(RagTool())
default_registry.register(ImageUnderstandingTool())
default_registry.register(ImageGenerationTool())
default_registry.register(ASRTool())
default_registry.register(TTSTool())
default_registry.register(VideoUnderstandingTool())
default_registry.register(VideoGenerationTool())
default_registry.register(PointCloudTool())
default_registry.register(MeshTool())
default_registry.register(NERFTool())

TOOL_REGISTRY: ToolRegistry = default_registry

__all__ = [
    "TOOL_REGISTRY",
    "ToolRegistry",
    "ChatTool",
    "CodeTool",
    "RagTool",
    "ImageUnderstandingTool",
    "ImageGenerationTool",
    "ASRTool",
    "TTSTool",
    "VideoUnderstandingTool",
    "VideoGenerationTool",
    "PointCloudTool",
    "MeshTool",
    "NERFTool",
]
