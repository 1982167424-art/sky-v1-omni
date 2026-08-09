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
    engine: Any | None = None,
    enable_engine: bool = False,
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

    # 推理引擎：当 enable_engine=True 或 engine 已传入时启用
    # 启用后，model 字段以 "sky-v1-" 开头的请求会走 SkyInferenceEngine
    if engine is None and enable_engine:
        try:
            from sky_v1.inference.engine import SkyInferenceEngine  # type: ignore
            from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig  # type: ignore

            _mini_cfg = SkyModelConfig(
                model_name="sky-v1-api-mini", hidden_dim=64, num_layers=2,
                num_heads=2, ffn_dim=128, max_seq_len=256, vocab_size=512,
                image_vocab_size=0, audio_vocab_size=0, video_vocab_size=0,
                three_d_vocab_size=0, modal=ModalConfig(), heads=HeadsConfig(),
                eos_token_id=2,
            )
            engine = SkyInferenceEngine(_mini_cfg, device="cpu", dtype="fp32")
        except Exception:
            engine = None

    app.state.rag_kb = rag_kb
    app.state.agent = agent
    app.state.engine = engine
    app.state.stats = {"req": 0, "errors": 0, "lat_sum": 0.0}

    try:
        add_middlewares(app)
    except Exception as e:
        log.warning("add_middlewares raised (non-fatal)", error=str(e))

    try:
        register_all_routes(app)
    except Exception as e:
        log.error("register_all_routes FAILED — app will have no routes!", error=str(e))
        raise

    # 根路径处理器：避免访问 / 时 404
    @app.get("/")
    async def root():
        return {
            "service": "sky-v1-omni API",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "health": "GET /health",
                "metrics": "GET /metrics",
                "chat_completions": "POST /v1/chat/completions",
                "completions": "POST /v1/completions",
                "embeddings": "POST /v1/embeddings",
                "rag_query": "POST /v1/rag/query",
                "rag_health": "GET /v1/rag/health",
                "rag_ingest": "POST /v1/rag/ingest",
                "images_generations": "POST /v1/images/generations",
                "audio_speech": "POST /v1/audio/speech",
                "audio_transcriptions": "POST /v1/audio/transcriptions",
                "videos_generations": "POST /v1/videos/generations",
                "3d_generations": "POST /v1/3d/generations",
                "agent_run": "POST /v1/agent/run",
                "agent_tools": "GET /v1/agent/tools",
                "search_web": "POST /v1/search/web",
                "reasoning_deep": "POST /v1/reasoning/deep",
            },
        }

    return app
