import ast
import importlib
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# CI 轻量化：当 torch / sentence-transformers 等重依赖不可用时，自动跳过
# 依赖它们的测试模块。
#
# 两阶段检测：
#   1) AST 静态扫描：直接 import torch/sentence_transformers/... 的模块 → 跳过
#   2) 间接依赖探测：提取测试文件的 `from sky_v1.X import ...`，尝试 import
#      这些 sky_v1 子模块；若因 torch 缺失而 ImportError → 跳过
# ---------------------------------------------------------------------------
_HEAVY_DEPS = ("torch", "sentence_transformers", "transformers", "onnxruntime")

_collect_ignore: list[str] = []
try:
    import torch  # noqa: F401
except ImportError:
    _tests_dir = Path(__file__).parent
    for _sub in ("unit", "integration", "e2e"):
        _sub_dir = _tests_dir / _sub
        if not _sub_dir.is_dir():
            continue
        for _f in _sub_dir.glob("test_*.py"):
            _rel = f"{_sub}/{_f.name}"
            try:
                _tree = ast.parse(_f.read_text(encoding="utf-8"), filename=str(_f))
            except Exception:
                continue

            # 阶段 1：直接 import 重依赖
            _needs_heavy = False
            _sky_imports: set[str] = set()
            for _node in ast.walk(_tree):
                if isinstance(_node, ast.Import):
                    for _alias in _node.names:
                        if any(_alias.name.startswith(_d) for _d in _HEAVY_DEPS):
                            _needs_heavy = True
                        if _alias.name.startswith("sky_v1"):
                            _sky_imports.add(_alias.name)
                elif isinstance(_node, ast.ImportFrom) and _node.module:
                    if any(_node.module.startswith(_d) for _d in _HEAVY_DEPS):
                        _needs_heavy = True
                    if _node.module.startswith("sky_v1"):
                        _sky_imports.add(_node.module)
                if _needs_heavy:
                    break
            if _needs_heavy:
                _collect_ignore.append(_rel)
                continue

            # 阶段 2：间接依赖（sky_v1.X 本身可能 import torch）
            for _mod in sorted(_sky_imports):
                try:
                    importlib.import_module(_mod)
                except ImportError:
                    _collect_ignore.append(_rel)
                    break
                except Exception:
                    pass  # 其他异常（如 TypeError）不阻塞，交给 pytest 正常报错

collect_ignore = _collect_ignore

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
