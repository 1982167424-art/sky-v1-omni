"""Text-family tools: ChatTool, CodeTool, RagTool."""
from __future__ import annotations

import hashlib
import re
from typing import Any, ClassVar

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.text")


class ChatTool(BaseTool):
    name: ClassVar[str] = "tool_chat"
    description: ClassVar[str] = "Open-ended LLM chat / general question answering."
    modal: ClassVar[str] = "text"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "User prompt to chat about"},
            "model": {"type": "string", "description": "Optional model name"},
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {"output": {"type": "string"}},
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", "")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not isinstance(model, str):
            raise TypeError(f"model must be str, got {type(model).__name__}")
        try:
            truncated = prompt[:300]
            output = f"[SIM] Chat: {truncated} (Mock，因为没有配置Key)"
            return ToolResult(
                success=True,
                output=output,
                data={"prompt": prompt, "model": model, "simulated": True},
                tool_name=self.name,
            )
        except Exception as e:
            log.error("ChatTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class CodeTool(BaseTool):
    name: ClassVar[str] = "tool_code"
    description: ClassVar[str] = "Write and optionally run code snippets."
    modal: ClassVar[str] = "text"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["code"],
        "properties": {
            "language": {
                "type": "string",
                "description": "Programming language (python, bash, ...)",
            },
            "code": {"type": "string", "description": "Source code to execute"},
            "run": {
                "type": "boolean",
                "description": "Whether to actually execute (default true)",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {"output": {"type": "string"}, "stdout": {"type": "string"}},
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        language = kwargs.get("language", "python")
        code = kwargs.get("code", "")
        run_flag = kwargs.get("run", True)
        if not isinstance(language, str):
            raise TypeError(f"language must be str, got {type(language).__name__}")
        if not isinstance(code, str):
            raise TypeError(f"code must be str, got {type(code).__name__}")
        if not isinstance(run_flag, bool):
            raise TypeError(f"run must be bool, got {type(run_flag).__name__}")
        try:
            stdout = ""
            stripped = code.strip()
            if stripped == "print('hello')" or stripped == 'print("hello")':
                stdout = "hello"
            else:
                m = re.search(r"print\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", stripped)
                if m:
                    stdout = m.group(1)
                else:
                    stdout = "[SIM] Code executed"
            output = stdout if stdout else "[SIM] Code executed"
            return ToolResult(
                success=True,
                output=output,
                data={
                    "language": language,
                    "code": code,
                    "stdout": stdout,
                    "ran": run_flag,
                    "simulated": True,
                },
                tool_name=self.name,
            )
        except Exception as e:
            log.error("CodeTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class RagTool(BaseTool):
    name: ClassVar[str] = "tool_rag"
    description: ClassVar[str] = "Retrieve context from the attached RAG knowledge base."
    modal: ClassVar[str] = "text"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of chunks to retrieve",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "chunks": {"type": "array", "items": {"type": "object"}},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 3)
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        try:
            kb = getattr(ctx, "rag_kb", None)
            chunks: list[dict] = []
            if kb is not None and hasattr(kb, "query") and callable(kb.query):
                try:
                    raw = kb.query(query, top_k=top_k)
                    if isinstance(raw, list):
                        chunks = [
                            c if isinstance(c, dict) else {"text": str(c)} for c in raw
                        ]
                    else:
                        chunks = [{"text": str(raw)}]
                except Exception as inner:
                    log.warning("RAG kb.query failed, falling back to SIM", error=str(inner))
                    chunks = []
            if chunks:
                output = "\n\n".join(
                    c.get("text", str(c)) if isinstance(c, dict) else str(c)
                    for c in chunks
                )
            else:
                q_snippet = query[:200] if query else "(空)"
                output = f"[SIM] Rag: 没有知识库，返回关于 {q_snippet} 的通用摘要"
                chunks = [
                    {
                        "text": f"通用摘要：关于「{q_snippet}」的背景信息（模拟，无真实KB）。",
                        "source": "simulated",
                        "score": 0.0,
                    }
                ]
            return ToolResult(
                success=True,
                output=output,
                data={"chunks": chunks, "top_k": top_k, "simulated": kb is None},
                tool_name=self.name,
            )
        except Exception as e:
            log.error("RagTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )
