"""sky_v1.agent.sky_agent: top-level orchestrator SkyAgent with sync + async APIs."""
from __future__ import annotations

from typing import Any

from sky_v1.agent.base import BaseTool, ToolContext, ToolRegistry, ToolResult
from sky_v1.agent.memory import LongTermMemory, ShortTermMemory
from sky_v1.agent.planner import PlannerLLM, ToolCallPlan
from sky_v1.agent.reflection import ReflectionEngine
from sky_v1.agent.tools import TOOL_REGISTRY
from sky_v1.utils.logging import get_logger

log = get_logger("agent.sky_agent")


def _try_load_knowledge_base() -> Any | None:
    """Attempt to import a KnowledgeBase from the (future) rag module.

    Returns None gracefully if the module doesn't exist yet so M1 bootstrapping
    doesn't crash.
    """
    try:
        from sky_v1.rag import KnowledgeBase  # type: ignore
    except Exception:
        return None
    try:
        return KnowledgeBase()
    except Exception as e:
        log.warning("KnowledgeBase auto-init failed, leaving as None", error=str(e))
        return None


class SkyAgent:
    """Main orchestrator: memory -> plan -> tool(s) -> reflection -> answer."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        planner: PlannerLLM | None = None,
        memory: ShortTermMemory | None = None,
        long_memory: LongTermMemory | None = None,
        reflection: ReflectionEngine | None = None,
        rag_kb: Any | None = None,
        config: dict | None = None,
    ) -> None:
        self.tools: ToolRegistry = tools if tools is not None else TOOL_REGISTRY
        if not isinstance(self.tools, ToolRegistry):
            raise TypeError(
                f"tools must be ToolRegistry, got {type(self.tools).__name__}"
            )

        self.planner: PlannerLLM = (
            planner if planner is not None else PlannerLLM()
        )
        if not isinstance(self.planner, PlannerLLM):
            raise TypeError(
                f"planner must be PlannerLLM, got {type(self.planner).__name__}"
            )

        self.memory: ShortTermMemory = (
            memory if memory is not None else ShortTermMemory()
        )
        if not isinstance(self.memory, ShortTermMemory):
            raise TypeError(
                f"memory must be ShortTermMemory, got {type(self.memory).__name__}"
            )

        self.long_memory: LongTermMemory = (
            long_memory if long_memory is not None else LongTermMemory()
        )
        if not isinstance(self.long_memory, LongTermMemory):
            raise TypeError(
                f"long_memory must be LongTermMemory, got {type(self.long_memory).__name__}"
            )

        self.reflection: ReflectionEngine = (
            reflection if reflection is not None else ReflectionEngine()
        )
        if not isinstance(self.reflection, ReflectionEngine):
            raise TypeError(
                f"reflection must be ReflectionEngine, got {type(self.reflection).__name__}"
            )

        self.rag_kb: Any = (
            rag_kb if rag_kb is not None else _try_load_knowledge_base()
        )

        self.config: dict = dict(config) if isinstance(config, dict) else {}

    def _build_ctx(self, session_id: str, user_id: str) -> ToolContext:
        if not isinstance(session_id, str):
            session_id = "default"
        if not isinstance(user_id, str):
            user_id = "anonymous"
        return ToolContext(
            session_id=session_id,
            user_id=user_id,
            rag_kb=self.rag_kb,
            config=dict(self.config),
            extra={},
        )

    @staticmethod
    def _concat_outputs(tool_calls: list[ToolResult]) -> str:
        parts: list[str] = []
        for tr in tool_calls:
            if isinstance(tr, ToolResult):
                parts.append(tr.output or "")
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return "\n---\n".join(p for p in parts if p is not None)

    def _execute_plan(
        self,
        ctx: ToolContext,
        plan: ToolCallPlan,
    ) -> tuple[list[ToolResult], str]:
        tool_calls: list[ToolResult] = []
        tool: BaseTool | None = self.tools.get(plan.tool_name)
        if tool is None:
            fallback = ToolResult(
                success=False,
                error=f"Unknown tool: {plan.tool_name!r}",
                tool_name=plan.tool_name,
            )
            tool_calls.append(fallback)
            return tool_calls, fallback.error or ""
        kwargs = plan.tool_kwargs if isinstance(plan.tool_kwargs, dict) else {}
        try:
            result: ToolResult = tool.run(ctx, **kwargs)
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=False,
                    error=f"Tool {plan.tool_name!r} did not return ToolResult",
                    tool_name=plan.tool_name,
                )
        except Exception as e:
            log.error("Tool run raised", tool_name=plan.tool_name, error=str(e))
            result = ToolResult(
                success=False,
                error=str(e),
                tool_name=plan.tool_name,
            )
        tool_calls.append(result)
        answer = self._concat_outputs(tool_calls)
        return tool_calls, answer

    async def _aexecute_plan(
        self,
        ctx: ToolContext,
        plan: ToolCallPlan,
    ) -> tuple[list[ToolResult], str]:
        tool_calls: list[ToolResult] = []
        tool: BaseTool | None = self.tools.get(plan.tool_name)
        if tool is None:
            fallback = ToolResult(
                success=False,
                error=f"Unknown tool: {plan.tool_name!r}",
                tool_name=plan.tool_name,
            )
            tool_calls.append(fallback)
            return tool_calls, fallback.error or ""
        kwargs = plan.tool_kwargs if isinstance(plan.tool_kwargs, dict) else {}
        try:
            result: ToolResult = await tool.arun(ctx, **kwargs)
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=False,
                    error=f"Tool {plan.tool_name!r} did not return ToolResult (async)",
                    tool_name=plan.tool_name,
                )
        except Exception as e:
            log.error("Tool arun raised", tool_name=plan.tool_name, error=str(e))
            result = ToolResult(
                success=False,
                error=str(e),
                tool_name=plan.tool_name,
            )
        tool_calls.append(result)
        answer = self._concat_outputs(tool_calls)
        return tool_calls, answer

    def step(
        self,
        user_message: str,
        *,
        session_id: str = "default",
        user_id: str = "anonymous",
        attachments: list[dict] | None = None,
        reflect_limit: int = 1,
    ) -> dict:
        if not isinstance(reflect_limit, int) or isinstance(reflect_limit, bool):
            raise TypeError(
                f"reflect_limit must be int, got {type(reflect_limit).__name__}"
            )
        if reflect_limit < 0:
            reflect_limit = 0
        if attachments is None:
            attachments = []
        try:
            self.memory.add("user", user_message if isinstance(user_message, str) else str(user_message))
            ctx = self._build_ctx(session_id, user_id)

            reflect_rounds = 0
            last_plan: ToolCallPlan | None = None
            last_tool_calls: list[ToolResult] = []
            last_answer: str = ""

            while True:
                history = self.memory.get()
                plan = self.planner.plan(ctx, user_message, attachments, history)
                last_plan = plan
                tool_calls, answer = self._execute_plan(ctx, plan)
                last_tool_calls = tool_calls
                last_answer = answer
                needs_rewrite, feedback = self.reflection.review(
                    plan, ctx, answer, tool_calls
                )
                if not needs_rewrite or reflect_rounds >= reflect_limit:
                    break
                reflect_rounds += 1
                self.memory.add(
                    "system",
                    f"[Reflection round {reflect_rounds}] {feedback}",
                )
                user_message = (
                    f"{user_message}\n\n[Reflection反馈: {feedback}]"
                )

            assistant_content = last_answer or "(空回复)"
            self.memory.add("assistant", assistant_content)

            plan_dict = last_plan.to_dict() if isinstance(last_plan, ToolCallPlan) else {
                "tool_name": "tool_chat",
                "tool_kwargs": {"prompt": user_message},
                "reasoning": "Fallback plan dict",
            }
            return {
                "answer": assistant_content,
                "tool_calls": last_tool_calls,
                "plan": plan_dict,
                "reflect_rounds": reflect_rounds,
                "session_id": ctx.session_id,
            }
        except Exception as e:
            log.error("SkyAgent.step crashed", error=str(e))
            err_answer = f"Agent发生内部错误已处理: {e}"
            try:
                self.memory.add("assistant", err_answer)
            except Exception:
                pass
            return {
                "answer": err_answer,
                "tool_calls": [],
                "plan": {
                    "tool_name": "tool_chat",
                    "tool_kwargs": {"prompt": user_message if isinstance(user_message, str) else str(user_message)},
                    "reasoning": "Fallback plan due to crash",
                },
                "reflect_rounds": 0,
                "session_id": session_id if isinstance(session_id, str) else "default",
            }

    async def astep(
        self,
        user_message: str,
        *,
        session_id: str = "default",
        user_id: str = "anonymous",
        attachments: list[dict] | None = None,
        reflect_limit: int = 1,
    ) -> dict:
        if not isinstance(reflect_limit, int) or isinstance(reflect_limit, bool):
            raise TypeError(
                f"reflect_limit must be int, got {type(reflect_limit).__name__}"
            )
        if reflect_limit < 0:
            reflect_limit = 0
        if attachments is None:
            attachments = []
        try:
            self.memory.add("user", user_message if isinstance(user_message, str) else str(user_message))
            ctx = self._build_ctx(session_id, user_id)

            reflect_rounds = 0
            last_plan: ToolCallPlan | None = None
            last_tool_calls: list[ToolResult] = []
            last_answer: str = ""

            while True:
                history = self.memory.get()
                plan = self.planner.plan(ctx, user_message, attachments, history)
                last_plan = plan
                tool_calls, answer = await self._aexecute_plan(ctx, plan)
                last_tool_calls = tool_calls
                last_answer = answer
                needs_rewrite, feedback = self.reflection.review(
                    plan, ctx, answer, tool_calls
                )
                if not needs_rewrite or reflect_rounds >= reflect_limit:
                    break
                reflect_rounds += 1
                self.memory.add(
                    "system",
                    f"[Reflection round {reflect_rounds}] {feedback}",
                )
                user_message = (
                    f"{user_message}\n\n[Reflection反馈: {feedback}]"
                )

            assistant_content = last_answer or "(空回复)"
            self.memory.add("assistant", assistant_content)

            plan_dict = last_plan.to_dict() if isinstance(last_plan, ToolCallPlan) else {
                "tool_name": "tool_chat",
                "tool_kwargs": {"prompt": user_message},
                "reasoning": "Fallback plan dict",
            }
            return {
                "answer": assistant_content,
                "tool_calls": last_tool_calls,
                "plan": plan_dict,
                "reflect_rounds": reflect_rounds,
                "session_id": ctx.session_id,
            }
        except Exception as e:
            log.error("SkyAgent.astep crashed", error=str(e))
            err_answer = f"Agent发生内部错误已处理: {e}"
            try:
                self.memory.add("assistant", err_answer)
            except Exception:
                pass
            return {
                "answer": err_answer,
                "tool_calls": [],
                "plan": {
                    "tool_name": "tool_chat",
                    "tool_kwargs": {"prompt": user_message if isinstance(user_message, str) else str(user_message)},
                    "reasoning": "Fallback plan due to crash",
                },
                "reflect_rounds": 0,
                "session_id": session_id if isinstance(session_id, str) else "default",
            }
