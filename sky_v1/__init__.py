"""sky_v1: 5-modal unified model + agent package.

Public surface:
    sky_v1.SkyModel, SkyTrainer, SkyInferenceEngine, SkySDK, SkyKnowledgeBase, SkyAgent
"""
from __future__ import annotations

__version__ = "0.1.0a1"

# --- model (M2) ---
MODEL_AVAILABLE = False
try:
    from .model.sky_model import SkyModel, build_model_from_config
    from .model.config import SkyModelConfig, load_config_from_yaml
    MODEL_AVAILABLE = True
except Exception:
    SkyModel = None  # type: ignore
    build_model_from_config = None  # type: ignore
    SkyModelConfig = None  # type: ignore
    load_config_from_yaml = None  # type: ignore

# --- training (M3) ---
TRAINING_AVAILABLE = False
try:
    from .training.trainer import SkyTrainer
    try:
        from .training.losses import KD3LayerLoss, InfoNCELoss, ReconMSELoss
    except Exception:
        KD3LayerLoss = None  # type: ignore
        InfoNCELoss = None  # type: ignore
        ReconMSELoss = None  # type: ignore
    try:
        from .training.distill import TeacherPool
    except Exception:
        TeacherPool = None  # type: ignore
    try:
        from .training.dpo import dpo_loss
    except Exception:
        dpo_loss = None  # type: ignore
    from .training.checkpoint import CheckpointManager
    TRAINING_AVAILABLE = True
except Exception:
    SkyTrainer = None  # type: ignore
    KD3LayerLoss = None  # type: ignore
    InfoNCELoss = None  # type: ignore
    ReconMSELoss = None  # type: ignore
    dpo_loss = None  # type: ignore
    TeacherPool = None  # type: ignore
    CheckpointManager = None  # type: ignore

# --- data ---
DATA_AVAILABLE = False
try:
    from .data.toy_generator import ToyDataGenerator
    from .data.datasets import Phase1Dataset, Phase2AlignDataset, Phase3DistillDataset
    DATA_AVAILABLE = True
except Exception:
    ToyDataGenerator = None  # type: ignore
    Phase1Dataset = None  # type: ignore
    Phase2AlignDataset = None  # type: ignore
    Phase3DistillDataset = None  # type: ignore

# --- rag (M1) ---
RAG_AVAILABLE = False
try:
    from .rag.knowledge_base import SkyKnowledgeBase
    RAG_AVAILABLE = True
except Exception:
    SkyKnowledgeBase = None  # type: ignore

# --- agent (M1) ---
AGENT_AVAILABLE = False
try:
    from .agent.sky_agent import SkyAgent
    AGENT_AVAILABLE = True
except Exception:
    SkyAgent = None  # type: ignore

# --- api (M1) ---
API_AVAILABLE = False
try:
    from .api.app import create_app
    API_AVAILABLE = True
except Exception:
    create_app = None  # type: ignore

# --- inference (M4) ---
INFERENCE_AVAILABLE = False
try:
    from .inference.engine import SkyInferenceEngine, GenerateResult
    from .inference.kv_cache import PagedKVCache
    INFERENCE_AVAILABLE = True
except Exception:
    SkyInferenceEngine = None  # type: ignore
    GenerateResult = None  # type: ignore
    PagedKVCache = None  # type: ignore

QUANT_AVAILABLE = False
try:
    from .inference.quant import W8A8Linear, W4A16Linear, quantize_model_, dequantize_model_
    QUANT_AVAILABLE = True
except Exception:
    W8A8Linear = None  # type: ignore
    W4A16Linear = None  # type: ignore
    quantize_model_ = None  # type: ignore
    dequantize_model_ = None  # type: ignore

LORA_AVAILABLE = False
try:
    from .model.lora import LoRALinear, mark_lora_targets_, merge_lora_, unload_lora_, get_lora_state_dict, load_lora_state_dict
    LORA_AVAILABLE = True
except Exception:
    LoRALinear = None  # type: ignore
    mark_lora_targets_ = None  # type: ignore
    merge_lora_ = None  # type: ignore
    unload_lora_ = None  # type: ignore
    get_lora_state_dict = None  # type: ignore
    load_lora_state_dict = None  # type: ignore

# --- sdk / cli (M5) ---
SDK_AVAILABLE = False
try:
    from .sdk.client import SkySDK
    SDK_AVAILABLE = True
except Exception:
    SkySDK = None  # type: ignore

CLI_AVAILABLE = False
try:
    from .cli.main import app as cli_app
    CLI_AVAILABLE = True
except Exception:
    cli_app = None  # type: ignore

# --- 搜索 / 深度推理 工具 (M5+) ---
SEARCH_TOOLS_AVAILABLE = False
try:
    from .agent.tools.search_tools import WebSearchTool, DeepReasoningTool
    SEARCH_TOOLS_AVAILABLE = True
except Exception:
    WebSearchTool = None  # type: ignore
    DeepReasoningTool = None  # type: ignore

__all__ = [
    "__version__",
    "SkyModel", "build_model_from_config", "SkyModelConfig", "load_config_from_yaml",
    "SkyTrainer", "KD3LayerLoss", "InfoNCELoss", "ReconMSELoss", "dpo_loss", "TeacherPool", "CheckpointManager",
    "ToyDataGenerator", "Phase1Dataset", "Phase2AlignDataset", "Phase3DistillDataset",
    "SkyKnowledgeBase",
    "SkyAgent",
    "create_app",
    "SkyInferenceEngine", "GenerateResult", "PagedKVCache",
    "W8A8Linear", "W4A16Linear", "quantize_model_", "dequantize_model_",
    "LoRALinear", "mark_lora_targets_", "merge_lora_", "unload_lora_", "get_lora_state_dict", "load_lora_state_dict",
    "SkySDK",
    "cli_app",
    "WebSearchTool", "DeepReasoningTool",
    "MODEL_AVAILABLE", "TRAINING_AVAILABLE", "DATA_AVAILABLE",
    "RAG_AVAILABLE", "AGENT_AVAILABLE", "API_AVAILABLE",
    "INFERENCE_AVAILABLE", "QUANT_AVAILABLE", "LORA_AVAILABLE", "SDK_AVAILABLE", "CLI_AVAILABLE",
    "SEARCH_TOOLS_AVAILABLE",
]
