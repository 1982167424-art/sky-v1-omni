"""search_routes: 毫秒级联网搜索 + 深度推理 端点。

与 Agent 路由相互独立：用户可直接调用 ``POST /v1/search/web`` 或
``POST /v1/reasoning/deep`` 而无需经过 Agent step 编排。
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from sky_v1.api.types import (
    DeepReasoningRequest,
    DeepReasoningResponse,
    WebSearchProviderStatus,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from sky_v1.agent.base import ToolContext

router = APIRouter()


def _get_search_tool(app):
    try:
        from sky_v1.agent.tools.search_tools import WebSearchTool
        return WebSearchTool()
    except Exception:
        return None


def _get_reasoning_tool(app):
    try:
        from sky_v1.agent.tools.search_tools import DeepReasoningTool
        return DeepReasoningTool()
    except Exception:
        return None


def _make_ctx(request: Request) -> ToolContext:
    rag_kb = getattr(request.app.state, "rag_kb", None)
    cfg = getattr(request.app.state, "config", None) or {}
    session_id = request.headers.get("x-sky-session-id", "api-call")
    user_id = request.headers.get("x-sky-user-id", "anonymous")
    return ToolContext(session_id=session_id, user_id=user_id, rag_kb=rag_kb, config=cfg)


@router.post("/search/web", response_model=WebSearchResponse)
async def search_web(
    request: Request,
    body: WebSearchRequest,
) -> WebSearchResponse:
    tool = _get_search_tool(request.app)
    ctx = _make_ctx(request)
    if tool is None:
        return WebSearchResponse(
            results=[],
            provider="unavailable",
            providers_used=[],
            provider_statuses=[
                WebSearchProviderStatus(provider="unavailable", available=False, reason="TOOL_LOAD_FAILED", count=0, latency_ms=0)
            ],
            status="FAILED",
            cached=False,
            simulated=False,
            latency_ms=0,
        )
    res = tool.run(
        ctx,
        query=body.query,
        num_results=body.num_results,
        providers=list(body.providers) if body.providers else None,
        freshness=body.freshness,
        skip_cache=body.skip_cache,
        allow_simulated=body.allow_simulated,
    )
    data = res.data or {}
    raw_results = data.get("results") or []
    results: list[WebSearchResult] = []
    for r in raw_results:
        if isinstance(r, dict):
            results.append(WebSearchResult(
                title=str(r.get("title", ""))[:300],
                url=str(r.get("url", ""))[:500],
                snippet=str(r.get("snippet", ""))[:800],
                provider=str(r.get("provider", ""))[:32],
            ))
    providers_used: list[str] = list(data.get("providers_used", []) or [])
    status_raw = str(data.get("status", "EMPTY"))
    if status_raw not in {"OK", "PARTIAL", "EMPTY", "FAILED"}:
        status_raw = "PARTIAL" if providers_used and results else "EMPTY"
    status_val: Literal["OK", "PARTIAL", "EMPTY", "FAILED"] = status_raw  # type: ignore[assignment]
    provider_statuses: list[WebSearchProviderStatus] = []
    for row in (data.get("provider_statuses") or []):
        if isinstance(row, dict):
            provider_statuses.append(WebSearchProviderStatus(
                provider=str(row.get("provider", "unknown")),
                available=bool(row.get("available", False)),
                reason=str(row.get("reason", "")),
                count=int(row.get("count", 0) or 0),
                latency_ms=int(row.get("latency_ms", 0) or 0),
            ))
    primary = providers_used[0] if providers_used else (data.get("provider") or "unknown")
    return WebSearchResponse(
        results=results,
        provider=str(primary),
        providers_used=providers_used,
        provider_statuses=provider_statuses,
        status=status_val,
        cached=bool(data.get("cached", False)),
        simulated=bool(data.get("simulated", not res.success)),
        latency_ms=int(getattr(res, "latency_ms", 0) or 0),
    )


@router.post("/reasoning/deep", response_model=DeepReasoningResponse)
async def reasoning_deep(
    request: Request,
    body: DeepReasoningRequest,
) -> DeepReasoningResponse:
    tool = _get_reasoning_tool(request.app)
    ctx = _make_ctx(request)
    if tool is None:
        return DeepReasoningResponse(
            plan=[f"{body.question}：①直接答复"],
            iterations=[],
            final_answer=f"[Tool 未加载] 您的问题：{body.question[:100]}",
            confidence=0.1,
            simulated=True,
            latency_ms=0,
        )
    res = tool.run(
        ctx,
        question=body.question,
        max_iterations=body.max_iterations,
        enable_web_search=body.enable_web_search,
        citations_needed=body.citations_needed,
    )
    data = res.data or {}
    return DeepReasoningResponse(
        plan=list(data.get("plan", []) or []),
        iterations=list(data.get("iterations", []) or []),
        final_answer=str(data.get("final_answer", res.output)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        simulated=bool(data.get("simulated", not res.success)),
        latency_ms=int(getattr(res, "latency_ms", 0) or 0),
    )
