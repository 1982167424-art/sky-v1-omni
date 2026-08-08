from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from sky_v1 import __version__
from sky_v1.utils.logging import get_logger, setup_root_logger

from .middleware import add_middlewares
from .routes import register_all_routes

log = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        setup_root_logger()
    except Exception:
        pass
    try:
        log.info(
            "sky-v1-omni API starting",
            version=__version__,
            config_dir=str(getattr(app.state, "config_dir", "")),
        )
    except Exception:
        pass
    yield
    try:
        log.info("sky-v1-omni API shutdown complete")
    except Exception:
        pass


def create_app(
    *,
    config_dir: str | Path = "./configs",
    rag_kb: Any | None = None,
    agent: Any | None = None,
) -> FastAPI:
    app = FastAPI(
        title="sky-v1-omni API",
        version=__version__,
        description="sky-v1: 5-modal (Text/Image/3D/Video/Audio) Omni Agent - OpenAI-compatible API with Modal extensions",
        lifespan=lifespan,
    )

    try:
        app.state.config_dir = Path(config_dir).resolve()
    except Exception:
        app.state.config_dir = Path("./configs").resolve()

    if rag_kb is None:
        try:
            from sky_v1.rag import KnowledgeBase  # type: ignore

            try:
                rag_cfg_path = app.state.config_dir / "rag" / "vector_db_chroma.yaml"
                rag_kb = KnowledgeBase(str(rag_cfg_path))
            except Exception:
                try:
                    rag_kb = KnowledgeBase()
                except Exception:
                    rag_kb = None
        except ImportError:
            rag_kb = None
        except Exception:
            rag_kb = None

    if agent is None:
        try:
            from sky_v1.agent import SkyAgent  # type: ignore

            try:
                agent = SkyAgent(rag_kb=rag_kb)
            except Exception:
                agent = None
        except ImportError:
            agent = None
        except Exception:
            agent = None

    app.state.rag_kb = rag_kb
    app.state.agent = agent
    app.state.stats = {"req": 0, "errors": 0, "lat_sum": 0.0}

    try:
        add_middlewares(app)
    except Exception as e:
        log.warning("add_middlewares raised (non-fatal)", error=str(e))

    try:
        register_all_routes(app)
    except Exception as e:
        log.warning("register_all_routes raised (non-fatal)", error=str(e))

    return app
