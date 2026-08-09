from __future__ import annotations

from fastapi import FastAPI

from sky_v1.utils.logging import get_logger

from . import agent_routes, chat_routes, health_routes, modal_routes, rag_routes, search_routes

log = get_logger("api.routes")


def register_all_routes(app: FastAPI) -> None:
    try:
        app.include_router(health_routes.router, tags=["health"])
    except Exception as e:
        log.warning("Failed to register health_routes", error=str(e))

    try:
        app.include_router(chat_routes.router, prefix="/v1", tags=["chat"])
    except Exception as e:
        log.warning("Failed to register chat_routes", error=str(e))

    try:
        app.include_router(rag_routes.router, prefix="/v1", tags=["rag"])
    except Exception as e:
        log.warning("Failed to register rag_routes", error=str(e))

    try:
        app.include_router(modal_routes.router, prefix="/v1", tags=["modal"])
    except Exception as e:
        log.warning("Failed to register modal_routes", error=str(e))

    try:
        app.include_router(agent_routes.router, prefix="/v1", tags=["agent"])
    except Exception as e:
        log.warning("Failed to register agent_routes", error=str(e))

    try:
        app.include_router(search_routes.router, prefix="/v1", tags=["search/reasoning"])
    except Exception as e:
        log.warning("Failed to register search_routes", error=str(e))

    log.info("All API routes registered")
