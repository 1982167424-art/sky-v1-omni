"""3D-family tools: PointCloudTool, MeshTool, NERFTool."""
from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.threed")


def _md5(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"_md5 expects str, got {type(s).__name__}")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class PointCloudTool(BaseTool):
    name: ClassVar[str] = "tool_3d_pointcloud"
    description: ClassVar[str] = "Generate a colored point cloud (PLY) from a prompt."
    modal: ClassVar[str] = "3d"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Object/scene description"},
            "n_points": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of points to sample",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "obj_url": {"type": "string"},
            "n_points": {"type": "integer"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        prompt = kwargs.get("prompt", "")
        n_points = kwargs.get("n_points", 16384)
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not isinstance(n_points, int) or isinstance(n_points, bool):
            raise TypeError(f"n_points must be int, got {type(n_points).__name__}")
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        try:
            h = _md5(prompt or "empty")
            data = {
                "obj_url": f"https://placeholder.local/sim-pc-{h}.ply",
                "n_points": n_points,
                "prompt": prompt,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output="Generated simulated point cloud (PLY)",
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("PointCloudTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class MeshTool(BaseTool):
    name: ClassVar[str] = "tool_3d_mesh"
    description: ClassVar[str] = "Generate a triangle mesh (OBJ or GLB) from a prompt."
    modal: ClassVar[str] = "3d"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Object description"},
            "format": {
                "type": "string",
                "description": "Mesh format: obj, glb, gltf",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "obj_url": {"type": "string"},
            "vertices": {"type": "integer"},
            "faces": {"type": "integer"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        prompt = kwargs.get("prompt", "")
        fmt = kwargs.get("format", "obj")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not isinstance(fmt, str):
            raise TypeError(f"format must be str, got {type(fmt).__name__}")
        safe_fmt = fmt.lower() if fmt else "obj"
        if safe_fmt not in {"obj", "glb", "gltf"}:
            safe_fmt = "obj"
        try:
            h = _md5(prompt or "empty")
            data = {
                "obj_url": f"https://placeholder.local/sim-mesh-{h}.{safe_fmt}",
                "vertices": 1234,
                "faces": 800,
                "format": safe_fmt,
                "prompt": prompt,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output="Generated simulated mesh (OBJ/GLB)",
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("MeshTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class NERFTool(BaseTool):
    name: ClassVar[str] = "tool_3d_nerf"
    description: ClassVar[str] = (
        "Train a tiny NeRF-style density field checkpoint from a prompt / views."
    )
    modal: ClassVar[str] = "3d"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Scene description"},
            "views": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of training views",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "ckpt_url": {"type": "string"},
            "views": {"type": "integer"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        prompt = kwargs.get("prompt", "")
        views = kwargs.get("views", 24)
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not isinstance(views, int) or isinstance(views, bool):
            raise TypeError(f"views must be int, got {type(views).__name__}")
        if views < 1:
            raise ValueError(f"views must be >= 1, got {views}")
        try:
            h = _md5(prompt or "empty")
            data = {
                "ckpt_url": f"https://placeholder.local/sim-nerf-{h}.pt",
                "views": views,
                "prompt": prompt,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output="Generated simulated NeRF density field",
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("NERFTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )
