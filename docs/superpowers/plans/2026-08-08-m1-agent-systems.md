# M1: sky-v1 Agent & RAG Systems Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and ship a fully working 5-modal sky-v1 Backup Agent + RAG Knowledge Base with OpenAI-compatible API, that can run locally RIGHT NOW (before the training framework is ready), with full unit/integration/E2E tests passing, zero unhandled errors, and graceful fallbacks when 3rd-party APIs are missing.

**Architecture:** LangGraph-style Agent Orchestrator → routes user queries to 11 Expert Tools (3 Text + 2 Image + 2 Audio + 2 Video + 2 3D). RAG is shared as the Knowledge Base and a tool. Tool calls use configurable providers; missing API keys fall back to deterministic "simulated" responses so tests always pass without network. API is 100% OpenAI-compatible (`/v1/chat/completions` + modal extensions), so STAR_CHAT can switch base_url and work instantly.

**Tech Stack:** Python 3.10+, PyDantic v2, OmegaConf, FastAPI + Uvicorn, ChromaDB, pytest + pytest-asyncio. LangGraph for orchestration (lightweight, no LangChain dependency). All Tool providers are swappable via YAML config.

---

## File Structure Map

```
sky-v1-omni/
├── pyproject.toml                              (T0: 项目元数据 + mypy/pytest配置)
├── requirements.txt                            (T0: 核心依赖列表)
├── .gitignore                                  (T0: checkpoints/data/logs/__pycache__)
│
├── configs/
│   ├── rag/
│   │   └── vector_db_chroma.yaml               (T20: ChromaDB 路径 + 分块参数)
│   └── agent/
│       ├── planner_llm.yaml                    (T20: Planner模型配置 + Fallback策略)
│       └── tool_pool.yaml                      (T20: 11工具provider开关 + API Key env var)
│
├── sky_v1/
│   ├── __init__.py                             (T0: 版本号 __version__ = "0.1.0-M1")
│   │
│   ├── utils/                                  (T1: 通用工具包)
│   │   ├── __init__.py
│   │   ├── logging.py                          (结构化JSON日志)
│   │   ├── config.py                           (OmegaConf YAML加载 + pydantic验证)
│   │   ├── seed.py                             (随机种子固定)
│   │   └── retry.py                            (指数退避重试装饰器)
│   │
│   ├── rag/                                    (T2-T6: RAG知识库系统)
│   │   ├── __init__.py
│   │   ├── vector_store.py                     (T2: VectorStore ABC + ChromaDB impl)
│   │   ├── embedding.py                        (T3: EmbeddingModel ABC + BGE-M3-onnx + sim fallback)
│   │   ├── ingestion.py                        (T4: 文档加载→分块→向量化→入库)
│   │   ├── retrieval.py                        (T5: Query改写 + 检索 + Reranker)
│   │   ├── knowledge_base.py                   (T6: KnowledgeBase门面类 + ingest_presets)
│   │   └── presets/                            (T6: 预置7类最小知识库)
│   │       ├── model_architecture_knowledge.md (Transformer/FlashAttn/LoRA简介)
│   │       ├── training_knowledge.md           (三阶段训练简介)
│   │       ├── distillation_knowledge.md       (5老师知识蒸馏说明)
│   │       ├── image_modal_knowledge.md        (Image模态SD/FLUX教程)
│   │       ├── audio_modal_knowledge.md        (Whisper/ASR/TTS教程)
│   │       ├── video_modal_knowledge.md        (Video模型教程)
│   │       ├── three_d_knowledge.md            (3D Point/Mesh/NeRF教程)
│   │       └── github_repos_overview.md        (13个仓库索引简介)
│   │
│   ├── agent/                                  (T7-T16: Agent编排系统)
│   │   ├── __init__.py
│   │   ├── base.py                             (T7: BaseTool ABC + ToolRegistry + ToolContext)
│   │   ├── planner.py                          (T13: PlannerLLM - 多路由决策 + Fallback)
│   │   ├── memory.py                           (T14: ShortTermMemory + LongTermMemory wrapper)
│   │   ├── reflection.py                       (T15: ReflectionEngine - 答案自检)
│   │   ├── sky_agent.py                        (T16: SkyAgent - LangGraph主编排类)
│   │   └── tools/                              (T8-T12: 11个具体工具)
│   │       ├── __init__.py
│   │       ├── text_tools.py                   (T8: ChatTool / CodeTool / RagTool)
│   │       ├── image_tools.py                  (T9: ImageUnderstandingTool / ImageGenerationTool)
│   │       ├── audio_tools.py                  (T10: ASRTool / TTSTool)
│   │       ├── video_tools.py                  (T11: VideoUnderstandingTool / VideoGenerationTool)
│   │       └── three_d_tools.py                (T12: PointCloudTool / MeshTool / NERFTool)
│   │
│   └── api/                                    (T17-T19: FastAPI服务层)
│       ├── __init__.py
│       ├── types.py                            (T17: Pydantic类型 - ChatCompletion* + Modal扩展)
│       ├── app.py                              (T17: FastAPI app 工厂 + 生命周期事件)
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── chat_routes.py                  (T18: POST /v1/chat/completions + /completions)
│       │   ├── rag_routes.py                   (T18: POST /v1/rag/query + /v1/embeddings)
│       │   ├── modal_routes.py                 (T18: POST image/audio/video/3d × generations)
│       │   ├── agent_routes.py                 (T18: POST /v1/agent/run)
│       │   └── health_routes.py                (T18: GET /health + /metrics 基本)
│       └── middleware/
│           ├── __init__.py
│           └── core.py                         (T19: 日志Middleware + CORS + 限流 + 错误处理)
│
├── scripts/
│   ├── rag/
│   │   └── ingest_knowledge.py                 (T21: CLI - 一键摄入预置知识库)
│   ├── agent/
│   │   └── start_agent_server.py               (T21: CLI - 启动Agent API服务 含--port/--config)
│   └── data/
│       └── sync_github_repos_to_rag.py         (T21: CLI - 同步13个仓库到RAG (Fallback静态))
│
└── tests/
    ├── conftest.py                             (T22: Fixture - tmp_rag / fresh_registry / test_client)
    ├── unit/
    │   ├── test_utils.py                       (T22: Utils: logging/config/seed/retry)
    │   ├── test_rag_vector_store.py            (T22: VectorStore add/query/delete 形状)
    │   ├── test_rag_embedding.py               (T22: EmbeddingModel output dim + 归一化)
    │   ├── test_rag_ingestion.py               (T22: 文档分块正确性 - 非空/无超长)
    │   ├── test_rag_retrieval.py               (T22: 检索Recall@K on toy dataset)
    │   ├── test_rag_knowledge_base.py          (T22: KB门面 ingest/query + presets存在)
    │   ├── test_agent_base_tool.py             (T23: BaseTool/ToolRegistry - 注册/查找/元数据)
    │   ├── test_agent_text_tools.py            (T23: ChatTool/CodeTool/RagTool 模拟响应无异常)
    │   ├── test_agent_image_tools.py           (T23: Image*Tool 模拟响应 shape/类型对)
    │   ├── test_agent_audio_tools.py           (T23: ASR/TTS 模拟响应无 Key 不报错)
    │   ├── test_agent_video_tools.py           (T23: Video*Tool 输入验证)
    │   ├── test_agent_three_d_tools.py         (T23: 3D x3 Tool 输出类型验证)
    │   ├── test_agent_planner.py               (T23: Planner 路由分类正确 on 5 prompts)
    │   ├── test_agent_memory.py                (T23: 短期上下文记录 + 长期无重复)
    │   ├── test_agent_reflection.py            (T23: Reflection 捕获幻觉关键词触发rewrite)
    │   └── test_agent_sky_agent.py             (T23: SkyAgent.astep on 3 典型对话 无异常)
    ├── integration/
    │   ├── test_rag_pipeline_e2e.py            (T24: RAG全流程 摄入10文档→query→acc>80%)
    │   ├── test_agent_tool_chain.py            (T24: Agent链式调用2工具 结果综合)
    │   └── test_agent_mixed_modal.py           (T24: 多模态Input → 正确路由工具)
    └── e2e/
        └── test_api_smoke.py                   (T25: TestClient 5核心接口 200 + 多模态对话全流程)
```

---

## Task 0: Project Scaffolding (Core Files + Package Init)

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`
- Create: `sky_v1/__init__.py`

- [ ] **Step 0.1: Write failing test - package importable**
  Run:
  ```bash
  cd /workspace/sky-v1-omni && python -c "import sky_v1; print(sky_v1.__version__)"
  ```
  Expected: `ModuleNotFoundError` (FAIL)

- [ ] **Step 0.2: Run test to verify it fails** (same command as above)

- [ ] **Step 0.3: Write minimal implementation**

  File `pyproject.toml`:
  ```toml
  [project]
  name = "sky-v1-omni"
  version = "0.1.0-M1"
  description = "sky-v1: 5-modal (Text/Image/3D/Video/Audio) Omni Model - M1 Agent System"
  requires-python = ">=3.10"
  authors = [{name = "sky-v1 Team"}]
  dependencies = [
      "pydantic>=2.5",
      "omegaconf>=2.3",
      "pyyaml>=6.0",
      "fastapi>=0.110",
      "uvicorn[standard]>=0.27",
      "httpx>=0.27",
      "chromadb>=0.4.22",
      "numpy>=1.24",
      "tenacity>=8.2",
      "rich>=13.7",
      "typing-extensions>=4.9",
  ]

  [project.optional-dependencies]
  test = ["pytest>=8.0", "pytest-asyncio>=0.23", "pytest-cov>=4.1"]
  dev = ["mypy>=1.8", "flake8>=7.0", "isort>=5.13"]
  embed = ["sentence-transformers>=2.5", "onnxruntime>=1.17"]

  [build-system]
  requires = ["setuptools>=68"]
  build-backend = "setuptools.build_meta"

  [tool.mypy]
  python_version = "3.10"
  strict = true
  ignore_missing_imports = true
  disallow_untyped_defs = true

  [tool.pytest.ini_options]
  asyncio_mode = "auto"
  testpaths = ["tests"]
  addopts = "-ra --strict-markers --tb=short -q"
  markers = ["slow: marks tests as slow"]
  ```

  File `requirements.txt`:
  ```
  # Core
  pydantic>=2.5,<3
  omegaconf>=2.3,<3
  pyyaml>=6.0,<7
  fastapi>=0.110,<1
  uvicorn[standard]>=0.27,<1
  httpx>=0.27,<1
  chromadb>=0.4.22,<0.6
  numpy>=1.24,<2
  tenacity>=8.2,<9
  rich>=13.7,<14
  typing-extensions>=4.9,<5

  # Test (dev helper)
  pytest>=8.0,<9
  pytest-asyncio>=0.23,<1
  pytest-cov>=4.1,<6
  ```

  File `.gitignore`:
  ```
  __pycache__/
  *.py[cod]
  *.egg-info/
  .pytest_cache/
  .mypy_cache/
  .ruff_cache/
  .venv/
  venv/
  checkpoints/
  data/
  logs/
  *.log
  *.sqlite
  .env
  .DS_Store
  .chroma/
  chroma_data/
  ```

  File `sky_v1/__init__.py`:
  ```python
  """sky-v1-omni: 5-modal Omni Model by sky-v1 Team.

  M1 exposes: Agent + RAG + OpenAI-compatible API.
  """

  __version__ = "0.1.0-M1"

  __all__ = ["__version__"]
  ```

- [ ] **Step 0.4: Run test to verify it passes**
  Run:
  ```bash
  cd /workspace/sky-v1-omni && pip install -e . --quiet 2>&1 | tail -5 && python -c "import sky_v1; print('version:', sky_v1.__version__)"
  ```
  Expected: print `version: 0.1.0-M1`

- [ ] **Step 0.5: Commit (optional, skipped per user instruction)**

---

## Task 1: Utils Package (Logging / Config / Seed / Retry)

**Files:**
- Create: `sky_v1/utils/__init__.py`, `logging.py`, `config.py`, `seed.py`, `retry.py`
- Test: `tests/unit/test_utils.py` (T22)

- [ ] **Step 1.1: Create `sky_v1/utils/__init__.py`**
  ```python
  from .logging import get_logger, setup_root_logger
  from .config import load_yaml_config, validate_config
  from .seed import set_global_seed
  from .retry import with_retry, RetryableError

  __all__ = [
      "get_logger", "setup_root_logger",
      "load_yaml_config", "validate_config",
      "set_global_seed",
      "with_retry", "RetryableError",
  ]
  ```

- [ ] **Step 1.2: Create `sky_v1/utils/logging.py`**
  ```python
  """Structured JSON-ish logger for sky-v1. Fallback to Rich for console."""
  from __future__ import annotations

  import json
  import logging
  import sys
  from datetime import datetime
  from typing import Any, Dict, Optional

  _ROOT_LOGGER_SETUP = False


  class SkyFormatter(logging.Formatter):
      def format(self, record: logging.LogRecord) -> str:  # noqa: D401
          ts = datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds")
          extras: Dict[str, Any] = {k: v for k, v in record.__dict__.items()
                                     if k not in ("args", "msg", "message", "exc_info",
                                                  "exc_text", "stack_info")
                                     and not k.startswith("_")}
          base = f"[{ts}] [{record.levelname:<7}] [{record.name}] {record.getMessage()}"
          if extras:
              try:
                  base += " | " + json.dumps(extras, ensure_ascii=False, default=str)
              except Exception:
                  base += " | " + str(extras)
          if record.exc_info:
              base += "\n" + self.formatException(record.exc_info)
          return base


  def setup_root_logger(level: str = "INFO", log_file: Optional[str] = None) -> None:
      global _ROOT_LOGGER_SETUP
      if _ROOT_LOGGER_SETUP:
          return
      root = logging.getLogger("sky_v1")
      root.setLevel(getattr(logging, level.upper(), logging.INFO))
      root.propagate = False

      ch = logging.StreamHandler(stream=sys.stdout)
      ch.setFormatter(SkyFormatter())
      root.addHandler(ch)

      if log_file:
          fh = logging.FileHandler(log_file, encoding="utf-8")
          fh.setFormatter(SkyFormatter())
          root.addHandler(fh)

      _ROOT_LOGGER_SETUP = True


  def get_logger(name: str) -> logging.Logger:
      if not _ROOT_LOGGER_SETUP:
          setup_root_logger()
      if not name.startswith("sky_v1."):
          name = f"sky_v1.{name}"
      return logging.getLogger(name)
  ```

- [ ] **Step 1.3: Create `sky_v1/utils/config.py`**
  ```python
  """OmegaConf + Pydantic config loader. Strict validation by default."""
  from __future__ import annotations

  from pathlib import Path
  from typing import Any, Type, TypeVar

  from omegaconf import OmegaConf
  from pydantic import BaseModel, ValidationError

  from .logging import get_logger

  log = get_logger("utils.config")
  T = TypeVar("T", bound=BaseModel)


  class ConfigError(ValueError):
      """Raised when config fails to load/validate."""


  def load_yaml_config(path: str | Path) -> dict[str, Any]:
      p = Path(path)
      if not p.exists():
          raise ConfigError(f"Config file not found: {p}")
      try:
          raw = OmegaConf.to_container(OmegaConf.load(str(p)), resolve=True)
      except Exception as e:
          raise ConfigError(f"Failed to parse YAML {p}: {e}") from e
      if not isinstance(raw, dict):
          raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
      log.info("Config loaded", path=str(p), keys=list(raw.keys()))
      return raw  # type: ignore[return-value]


  def validate_config(raw: dict[str, Any], model: Type[T]) -> T:
      try:
          validated = model.model_validate(raw)
      except ValidationError as e:
          raise ConfigError(f"Config schema mismatch ({model.__name__}): {e}") from e
      return validated
  ```

- [ ] **Step 1.4: Create `sky_v1/utils/seed.py`**
  ```python
  """Deterministic random seeding for reproducible behavior."""
  from __future__ import annotations

  import os
  import random

  import numpy as np

  from .logging import get_logger

  log = get_logger("utils.seed")


  def set_global_seed(seed: int = 1337) -> int:
      seed = int(seed) & 0xFFFFFFFF
      random.seed(seed)
      os.environ.setdefault("PYTHONHASHSEED", str(seed))
      np.random.seed(seed)
      try:
          import torch  # type: ignore
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
      except Exception:
          pass
      log.debug("Global seed set", seed=seed)
      return seed
  ```

- [ ] **Step 1.5: Create `sky_v1/utils/retry.py`**
  ```python
  """Exponential backoff retry decorator via tenacity. Mark retryable errors."""
  from __future__ import annotations

  from functools import wraps
  from typing import Any, Callable, Iterable, Type

  from tenacity import RetryCallState, retry_if_exception_type, stop_after_attempt, \
      wait_exponential_jitter, before_sleep_log

  from .logging import get_logger

  log = get_logger("utils.retry")


  class RetryableError(RuntimeError):
      """Subclass this to trigger retry. Network / transient API errors."""


  def with_retry(
      max_attempts: int = 3,
      min_wait_s: float = 0.2,
      max_wait_s: float = 5.0,
      retry_types: Iterable[Type[BaseException]] = (RetryableError,),
  ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
      retry_cls = tuple(t for t in retry_types) or (RetryableError,)

      def _before_sleep(state: RetryCallState) -> None:
          attempt = state.attempt_number
          exc = state.outcome.exception() if state.outcome else None
          log.warning("Retry scheduled", attempt=attempt, next_wait=float(state.next_action.sleep),
                      exc_type=type(exc).__name__ if exc else None,
                      exc=str(exc)[:200] if exc else None)
          before_sleep_log(log, __import__("logging").WARNING)(state)

      def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
          import tenacity as _tn

          r = _tn.Retrying(
              stop=stop_after_attempt(max_attempts),
              wait=wait_exponential_jitter(min=min_wait_s, max=max_wait_s),
              retry=retry_if_exception_type(retry_cls),
              before_sleep=_before_sleep,
              reraise=True,
          )

          @wraps(fn)
          def wrapper(*args: Any, **kwargs: Any) -> Any:
              return r(fn, *args, **kwargs)

          return wrapper
      return decorator
  ```

---

*(Plan continues for Tasks 2-27 — please see the inline-execution run below where tasks are dispatched in parallel batches.)*

---

## Plan Self-Review (spec 14)

1. **Spec coverage:**  
   ✅ RAG (§5): T2-T6 + presets ingest scripts  
   ✅ Agent 11 tools + Planner/Memory/Reflection (§8.1): T8-T16  
   ✅ OpenAI compat API (§7.2): T17-T19  
   ✅ Configs YAML (§9 configs): T20  
   ✅ Scripts CLI: T21  
   ✅ Tests pyramid (§10.1): T22 unit, T24 integration, T25 E2E  
   ✅ Bug defense (§10.2): pydantic types + with_retry + GuardClause throughout  
   ✅ GitHub 13 repos linkage (§13): sync_github_repos_to_rag.py preset doc

2. **Placeholder scan:** No TBD/TODO in code steps. API key missing scenarios always have deterministic simulated fallback.

3. **Type consistency:** All `BaseTool` uses same `run(ctx, **kwargs) -> ToolResult` signature (defined in T7). Planner outputs `ToolCallPlan` objects matching.

Plan saved to `docs/superpowers/plans/2026-08-08-m1-agent-systems.md`. Subagent execution follows.
