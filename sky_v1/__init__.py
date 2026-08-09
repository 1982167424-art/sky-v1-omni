"""sky-v1-omni: 5-modal Omni Model.

Milestone 1 (0.1.0-M1): Backup Agent + RAG Knowledge Base + OpenAI-compatible API.
Share RAG + API contract with the training framework (Milestone 2+).
"""

__version__ = "0.1.0a1"

__all__ = [
    "__version__",
    "MODEL_AVAILABLE",
    "TRAINING_AVAILABLE",
    "DATA_AVAILABLE",
    "SkyModel", "SkyModelConfig", "ModalConfig", "HeadsConfig",
    "UniTransformerBackbone", "UniTransformerLayer", "build_model_from_config", "load_config_from_yaml",
    "SkyTrainer", "CheckpointManager", "MetricsLogger",
    "KD3LayerLoss", "InfoNCELoss", "TeacherPool",
    "ToyDataGenerator", "Phase1Dataset", "Phase2AlignDataset", "Phase3DistillDataset", "SkyDataCollator",
]

try:
    from .model import (
        SkyModel, SkyModelConfig, ModalConfig, HeadsConfig,
        UniTransformerBackbone, UniTransformerLayer, build_model_from_config,
        load_config_from_yaml,
    )
    from . import model as _model
    MODEL_AVAILABLE = True
except Exception as e:
    MODEL_AVAILABLE = False
    _MODEL_IMPORT_ERROR = str(e)

try:
    from .training import (
        SkyTrainer, CheckpointManager, MetricsLogger,
        KD3LayerLoss, InfoNCELoss, TeacherPool,
    )
    from . import training as _training
    TRAINING_AVAILABLE = True
except Exception as e:
    TRAINING_AVAILABLE = False
    _TRAINING_IMPORT_ERROR = str(e)

try:
    from .data import (
        ToyDataGenerator, Phase1Dataset, Phase2AlignDataset, Phase3DistillDataset, SkyDataCollator,
    )
    from . import data as _data
    DATA_AVAILABLE = True
except Exception as e:
    DATA_AVAILABLE = False
    _DATA_IMPORT_ERROR = str(e)
