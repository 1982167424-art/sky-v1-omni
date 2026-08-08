"""sky_v1.agent.base: Core abstractions for the tool subsystem.

Provides ToolContext, ToolResult, RetryableProviderError, BaseTool (ABC),
and ToolRegistry with strong contracts (types + guard clauses).
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from sky_v1.utils.logging import get_logger
from sky_v1.utils.retry import RetryableError

log = get_logger("agent.base")


class ToolContext(BaseModel):
    """Per-invocation context passed to every tool run."""

    session_id: str = "default"
    user_id: str = "anonymous"
    rag_kb: Any | None = None
    config: dict = {}
    extra: dict = {}


class ToolResult(BaseModel):
    """Standardised tool execution result.

    success=False implies error must be non-empty (enforced by factory helpers).
    """

    success: bool
    output: str = ""
    data: dict = {}
    error: str | None = None
    tool_name: str = ""
    latency_ms: int = 0
    metadata: dict = {}

    model_config = ConfigDict(extra="ignore")


class RetryableProviderError(RetryableError):
    """Third-party provider API failure that is safe to retry (5xx, timeout, 429)."""


class BaseTool(BaseModel, ABC):
    """Abstract base for every tool in the registry.

    Subclasses MUST override the four ClassVar declarations below, and implement
    `run`.  `arun` delegates to `run` synchronously inside a thread by default;
    tools with native async providers should override for efficiency.
    """

    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = "Base tool"
    modal: ClassVar[str] = "text"
    inputs_schema: ClassVar[dict] = {}
    outputs_schema: ClassVar[dict] = {}

    model_config = ConfigDict(extra="ignore")

    @abstractmethod
    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        ...

    async def arun(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.run(ctx, **kwargs)
        )

    def _provider_missing_result(
        self, provider_name: str, note: str = ""
    ) -> ToolResult:
        """Return a SIMULATED success result when a provider key is missing.

        This guarantees tests and dry-runs pass without API keys.  The output
        clearly marks the result as simulated and data carries a sensible fake
        structure.
        """
        if not isinstance(provider_name, str):
            raise TypeError(
                f"provider_name must be str, got {type(provider_name).__name__}"
            )
        if not isinstance(note, str):
            raise TypeError(f"note must be str, got {type(note).__name__}")
        header = f"[SIMULATED] {self.name}: 未配置 {provider_name}，返回模拟结果"
        output = header if not note else f"{header}。{note}"
        return ToolResult(
            success=True,
            output=output,
            data={"simulated": True, "provider": provider_name, "note": note},
            tool_name=self.name,
            metadata={"simulated": True},
        )


class ToolRegistry:
    """A simple in-memory registry mapping unique tool names -> BaseTool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"ToolRegistry.register expects BaseTool instance, "
                f"got {type(tool).__name__}"
            )
        if tool.name in self._tools:
            log.warning(
                "Overwriting existing tool registration",
                tool_name=tool.name,
                previous_type=type(self._tools[tool.name]).__name__,
                new_type=type(tool).__name__,
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        return self._tools.get(name)

    def list(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "modal": t.modal,
                "inputs": t.inputs_schema,
                "outputs": t.outputs_schema,
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
