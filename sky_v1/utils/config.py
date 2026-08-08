"""OmegaConf YAML loader + Pydantic v2 strict validation.

Usage:
    raw = load_yaml_config("configs/agent/planner_llm.yaml")
    cfg = validate_config(raw, PlannerLLMConfig)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Type, TypeVar

from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from .logging import get_logger

log = get_logger("utils.config")
T = TypeVar("T", bound=BaseModel)


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or schema-mismatched."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p.resolve()}")
    if not p.is_file():
        raise ConfigError(f"Config path is not a file: {p.resolve()}")
    try:
        cfg = OmegaConf.load(str(p))
        raw = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML at {p.resolve()}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top-level YAML must be a mapping (object), got {type(raw).__name__}"
        )
    log.debug("Config loaded", path=str(p.resolve()), keys=list(raw.keys()))
    return raw  # type: ignore[return-value]


def validate_config(raw: dict[str, Any], model: Type[T]) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"validate_config expects dict, got {type(raw).__name__}")
    try:
        validated = model.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(
            f"Config failed Pydantic validation against {model.__name__}:\n{e}"
        ) from e
    return validated
