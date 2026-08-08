"""Video-family tools: VideoUnderstandingTool, VideoGenerationTool."""
from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.video")


def _md5(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"_md5 expects str, got {type(s).__name__}")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class VideoUnderstandingTool(BaseTool):
    name: ClassVar[str] = "tool_video_understand"
    description: ClassVar[str] = "Summarize a video and answer questions about it."
    modal: ClassVar[str] = "video"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["video_url"],
        "properties": {
            "video_url": {"type": "string", "description": "Video URL to analyze"},
            "prompt": {
                "type": "string",
                "description": "Question to answer about the video",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "frames_n": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        video_url = kwargs.get("video_url", "")
        prompt = kwargs.get("prompt", "")
        if not isinstance(video_url, str):
            raise TypeError(f"video_url must be str, got {type(video_url).__name__}")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        try:
            output = (
                f"[SIM] Video Understand: 视频包含一个人在走路，背景是公园。回答 {prompt}"
            )
            data = {
                "frames_n": 10,
                "tags": ["park", "walking"],
                "video_url": video_url,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output=output,
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("VideoUnderstandingTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class VideoGenerationTool(BaseTool):
    name: ClassVar[str] = "tool_video_generate"
    description: ClassVar[str] = "Generate a short video clip from a text prompt."
    modal: ClassVar[str] = "video"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Video description"},
            "duration_s": {
                "type": "number",
                "description": "Duration in seconds",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "video_url": {"type": "string"},
            "duration_s": {"type": "number"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        prompt = kwargs.get("prompt", "")
        duration_s = kwargs.get("duration_s", 5)
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool):
            raise TypeError(
                f"duration_s must be int/float, got {type(duration_s).__name__}"
            )
        try:
            h = _md5(prompt or "empty")
            data = {
                "video_url": f"https://placeholder.local/sim-video-{h}.mp4",
                "duration_s": duration_s,
                "prompt": prompt,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output="Generated simulated video",
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("VideoGenerationTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )
