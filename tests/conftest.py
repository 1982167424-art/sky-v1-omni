import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sky_v1.utils.logging import setup_root_logger
from sky_v1.rag.vector_store import InMemoryStore
from sky_v1.rag.embedding import SimEmbeddingFallback
from sky_v1.rag.knowledge_base import KnowledgeBase
from sky_v1.agent.base import ToolRegistry, ToolContext
from sky_v1.agent.tools import (
    ChatTool,
    CodeTool,
    RagTool,
    ImageUnderstandingTool,
    ImageGenerationTool,
    ASRTool,
    TTSTool,
    VideoUnderstandingTool,
    VideoGenerationTool,
    PointCloudTool,
    MeshTool,
    NERFTool,
)
from sky_v1.agent.planner import PlannerLLM
from sky_v1.agent.sky_agent import SkyAgent


def pytest_configure(config):
    setup_root_logger("DEBUG")


@pytest.fixture(autouse=True)
def _monkeypatch_env(monkeypatch):
    env_keys_to_clear = [
        "OPENAI_API_KEY",
        "AZURE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "STABILITY_API_KEY",
        "REPLICATE_API_TOKEN",
        "HUGGINGFACE_API_KEY",
        "COHERE_API_KEY",
        "PINECONE_API_KEY",
        "CHROMA_PERSIST_DIR",
    ]
    for k in env_keys_to_clear:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SKY_V1_TEST_MODE", "1")


@pytest.fixture
def tmp_rag_store():
    return InMemoryStore()


@pytest.fixture
def tmp_embedder():
    return SimEmbeddingFallback()


@pytest.fixture
def fresh_registry():
    r = ToolRegistry()
    r.register(ChatTool())
    r.register(CodeTool())
    r.register(RagTool())
    r.register(ImageUnderstandingTool())
    r.register(ImageGenerationTool())
    r.register(ASRTool())
    r.register(TTSTool())
    r.register(VideoUnderstandingTool())
    r.register(VideoGenerationTool())
    r.register(PointCloudTool())
    r.register(MeshTool())
    r.register(NERFTool())
    return r


@pytest.fixture
def fresh_kb():
    return KnowledgeBase(store=InMemoryStore(), embedder=SimEmbeddingFallback())


@pytest.fixture
def preset_kb(fresh_kb):
    fresh_kb.ingest_presets()
    return fresh_kb


@pytest.fixture
def fresh_agent(fresh_registry):
    return SkyAgent(tools=fresh_registry, planner=PlannerLLM(), rag_kb=None)


@pytest.fixture
def test_client(fresh_kb, fresh_agent):
    from fastapi.testclient import TestClient
    from sky_v1.api.app import create_app

    app = create_app(rag_kb=fresh_kb, agent=fresh_agent)
    return TestClient(app)
