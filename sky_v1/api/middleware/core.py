from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from sky_v1.utils.logging import get_logger

log = get_logger("api.middleware")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())
        try:
            request.state.req_id = req_id
        except Exception:
            pass
        try:
            response = await call_next(request)
        except Exception:
            raise
        try:
            response.headers["X-Request-ID"] = req_id
        except Exception:
            pass
        return response


def _get_req_id(request: Request) -> str:
    try:
        return getattr(request.state, "req_id", "unknown")
    except Exception:
        return "unknown"


def _inc_stats(app: FastAPI, key: str, amt: int = 1) -> None:
    try:
        stats = getattr(app.state, "stats", None)
        if stats is None:
            return
        stats[key] = stats.get(key, 0) + amt
    except Exception:
        pass


def _add_latency(app: FastAPI, latency_ms: float) -> None:
    try:
        stats = getattr(app.state, "stats", None)
        if stats is None:
            return
        stats["lat_sum"] = stats.get("lat_sum", 0.0) + latency_ms
    except Exception:
        pass


def add_middlewares(app: FastAPI) -> None:
    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception as e:
        log.warning("Failed to add CORS middleware", error=str(e))

    try:
        app.add_middleware(RequestIdMiddleware)
    except Exception as e:
        log.warning("Failed to add RequestId middleware", error=str(e))

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> Response:
        _inc_stats(request.app, "errors")
        req_id = _get_req_id(request)
        msg = str(exc)[:500] if exc else "Unknown error"
        log.error("Unhandled exception", req_id=req_id, error=msg, exc_type=type(exc).__name__)
        try:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "server_error",
                        "message": msg,
                        "req_id": req_id,
                    }
                },
            )
        except Exception as e2:
            log.warning("Exception handler itself failed", error=str(e2))
            return Response(
                content='{"error":{"type":"server_error","message":"fatal"}}',
                status_code=500,
                media_type="application/json",
            )

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        req_id = "unknown"
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            status = 500
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            try:
                req_id = _get_req_id(request)
            except Exception:
                req_id = "unknown"
            try:
                _inc_stats(request.app, "req")
                _add_latency(request.app, latency_ms)
            except Exception:
                pass
            try:
                log.info(
                    "access",
                    method=method,
                    path=path,
                    status=status,
                    latency_ms=round(latency_ms, 2),
                    req_id=req_id,
                )
            except Exception:
                pass
