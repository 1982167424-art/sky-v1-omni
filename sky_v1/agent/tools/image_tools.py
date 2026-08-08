"""Image-family tools: ImageUnderstandingTool, ImageGenerationTool."""
from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.image")


def _md5(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"_md5 expects str, got {type(s).__name__}")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class ImageUnderstandingTool(BaseTool):
    name: ClassVar[str] = "tool_image_understand"
    description: ClassVar[str] = (
        "Describe / detect objects in an image and answer questions about it."
    )
    modal: ClassVar[str] = "image"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["image_url"],
        "properties": {
            "image_url": {"type": "string", "description": "URL of image to analyze"},
            "prompt": {
                "type": "string",
                "description": "Question to answer about the image",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "boxes": {"type": "array"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        image_url = kwargs.get("image_url", "")
        prompt = kwargs.get("prompt", "")
        if not isinstance(image_url, str):
            raise TypeError(f"image_url must be str, got {type(image_url).__name__}")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        try:
            output = (
                f"[SIM] Image Understand: 识别到图中包含一只猫和一个桌子。回答 {prompt}"
            )
            data = {
                "boxes": [
                    {"label": "cat", "xyxy": [10, 20, 180, 260], "score": 0.95},
                    {"label": "table", "xyxy": [200, 150, 500, 400], "score": 0.92},
                ],
                "tags": ["cat", "table"],
                "image_url": image_url,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output=output,
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("ImageUnderstandingTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class ImageGenerationTool(BaseTool):
    name: ClassVar[str] = "tool_image_generate"
    description: ClassVar[str] = "Generate an image from a text prompt."
    modal: ClassVar[str] = "image"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Image description"},
            "size": {
                "type": "string",
                "description": "Output size, e.g. 1024x1024",
            },
            "steps": {"type": "integer", "description": "Sampling steps"},
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "image_url": {"type": "string"},
            "size": {"type": "string"},
            "prompt": {"type": "string"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        prompt = kwargs.get("prompt", "")
        size = kwargs.get("size", "1024x1024")
        steps = kwargs.get("steps", 20)
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not isinstance(size, str):
            raise TypeError(f"size must be str, got {type(size).__name__}")
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise TypeError(f"steps must be int, got {type(steps).__name__}")
        try:
            h = _md5(prompt or "empty")
            data = {
                "image_url": f"https://placeholder.local/sim-{h}.png",
                "size": size,
                "prompt": prompt,
                "steps": steps,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output="Generated simulated image",
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("ImageGenerationTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )
