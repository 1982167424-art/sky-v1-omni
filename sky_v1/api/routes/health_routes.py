from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from sky_v1.api.types import HealthResponse, MetricsResponse

router = APIRouter()

_start_time = time.perf_counter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    rag_count = 0
    agent_ok = False
    try:
        rag_kb = getattr(request.app.state, "rag_kb", None)
        if rag_kb is not None:
            try:
                rag_count = int(rag_kb.count())
            except Exception:
                rag_count = 0
    except Exception:
        rag_count = 0
    try:
        agent = getattr(request.app.state, "agent", None)
        agent_ok = agent is not None
    except Exception:
        agent_ok = False
    uptime_s = time.perf_counter() - _start_time
    return HealthResponse(
        status="ok",
        uptime_s=round(uptime_s, 3),
        rag_count=rag_count,
        agent_ok=agent_ok,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    req_total = 0
    err_total = 0
    avg_lat = 0.0
    try:
        stats: dict[str, Any] | None = getattr(request.app.state, "stats", None)
        if stats is not None:
            req_total = int(stats.get("req", 0))
            err_total = int(stats.get("errors", 0))
            lat_sum = float(stats.get("lat_sum", 0.0))
            if req_total > 0:
                avg_lat = lat_sum / req_total
    except Exception:
        pass
    components: dict[str, Any] = {}
    try:
        rag_kb = getattr(request.app.state, "rag_kb", None)
        components["rag_kb"] = {"configured": rag_kb is not None}
    except Exception:
        components["rag_kb"] = {"configured": False}
    try:
        agent = getattr(request.app.state, "agent", None)
        components["agent"] = {"configured": agent is not None}
    except Exception:
        components["agent"] = {"configured": False}
    return MetricsResponse(
        requests_total=req_total,
        errors_total=err_total,
        avg_latency_ms=round(avg_lat, 3),
        components=components,
    )
