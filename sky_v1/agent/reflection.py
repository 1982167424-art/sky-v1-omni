"""sky_v1.agent.reflection: simple hallucination / low-quality answer review."""
from __future__ import annotations

from typing import Any

from sky_v1.agent.base import ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.reflection")


class ReflectionEngine:
    """Lightweight post-exec review: hallucination keyword scan + length sanity.

    Returns (needs_rewrite, feedback) so the orchestrator can decide whether to
    re-plan + re-run the tool(s).
    """

    HALLUCINATION_KEYWORDS: tuple[str, ...] = (
        "我确定100%不存在",
        "我亲眼看到",
        "绝对没有任何",
        "这是官方唯一指定",
    )

    SHORT_OK_EXACT: frozenset[str] = frozenset(
        {"ok", "done", "好的", "完成", "ok.", "done.", "ok!", "done!"}
    )

    def review(
        self,
        plan: Any,
        ctx: ToolContext,
        raw_answer: str,
        tool_calls_seen: list[ToolResult],
    ) -> tuple[bool, str]:
        if not isinstance(ctx, ToolContext):
            log.warning("ReflectionEngine.review: bad ctx type")
        if not isinstance(tool_calls_seen, list):
            tool_calls_seen = []
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer) if raw_answer is not None else ""

        reasons: list[str] = []

        answer = raw_answer
        lowered = answer.lower().strip()

        for kw in self.HALLUCINATION_KEYWORDS:
            if kw and kw in answer:
                reasons.append(f"检测到幻觉/硬断言关键词: {kw!r}")
                break

        if len(answer.strip()) < 10 and lowered not in self.SHORT_OK_EXACT:
            reasons.append(
                f"回答过短（{len(answer.strip())} chars）且不是明确的 OK/Done"
            )

        any_tool_failed = False
        for tr in tool_calls_seen:
            if isinstance(tr, ToolResult) and not tr.success:
                any_tool_failed = True
                break
        if any_tool_failed:
            reasons.append("有工具调用返回 success=False，建议重试/重写")

        needs_rewrite = len(reasons) > 0
        if needs_rewrite:
            bullets = "；".join(reasons)
            feedback = (
                f"需要重写答案：{bullets}。建议：补充真实上下文，去掉绝对化断言，"
                f"并让回答更完整（>= 10 chars 或明确 OK）。"
            )
        else:
            feedback = "OK"
        return needs_rewrite, feedback
