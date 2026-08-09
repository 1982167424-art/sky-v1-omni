from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator

class ModalConfig(BaseModel):
    vocab_size: int = 128000
    modal_dim: int = 2048
    modal_id: int = 0
    patch_size: int = 16
    image_size: int = 224
    in_channels: int = 3
    mel_bins: int = 128
    subsample: int = 4
    num_frames: int = 8
    frame_size: int = 224
    num_points: int = 8192
    point_dim: int = 6
    mesh_vertices: int = 2048

class HeadsConfig(BaseModel):
    vocab_size: int = 128000
    out_channels: int = 3
    patch_size: int = 16
    mel_bins: int = 128
    num_frames: int = 8
    num_points: int = 8192
    point_dim: int = 3
    mesh_vertices: int = 2048

class SkyModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = "sky-v1-1B"
    hidden_size: int = Field(2048, ge=64)
    num_hidden_layers: int = Field(1, ge=1, le=128)
    num_attention_heads: int = Field(8, ge=1)
    intermediate_size: int = Field(2048, ge=64)
    vocab_size: int = Field(1000, ge=128)
    max_position_embeddings: int = Field(512, ge=32)
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "swiglu"
    rope_theta: float = 10000.0
    modal_types: list[str] = Field(default_factory=lambda: ["text","image","audio","video","three_d"])
    modal: Any = Field(default_factory=dict)
    heads: Any = Field(default_factory=dict)

    image_vocab_size: int = 0
    audio_vocab_size: int = 0
    video_vocab_size: int = 0
    three_d_vocab_size: int = 0
    model_name_: Optional[str] = Field(None, alias="model_name")
    hidden_dim_: Optional[int] = Field(None, alias="hidden_dim")
    num_layers_: Optional[int] = Field(None, alias="num_layers")
    num_heads_: Optional[int] = Field(None, alias="num_heads")
    ffn_dim_: Optional[int] = Field(None, alias="ffn_dim")
    max_seq_len_: Optional[int] = Field(None, alias="max_seq_len")
    eos_token_id_: Optional[int] = Field(None, alias="eos_token_id")

    @model_validator(mode="after")
    def _compat_alias_apply(self) -> "SkyModelConfig":
        if self.model_name_ is not None:
            self.name = self.model_name_
        if self.hidden_dim_ is not None:
            self.hidden_size = self.hidden_dim_
        if self.num_layers_ is not None:
            self.num_hidden_layers = self.num_layers_
        if self.num_heads_ is not None:
            self.num_attention_heads = self.num_heads_
        if self.ffn_dim_ is not None:
            self.intermediate_size = self.ffn_dim_
        if self.max_seq_len_ is not None:
            self.max_position_embeddings = self.max_seq_len_
        return self

    @model_validator(mode="after")
    def _compat_modal_heads(self) -> "SkyModelConfig":
        if isinstance(self.modal, ModalConfig):
            single = self.modal
            self.modal = {k: single for k in ["text","image","audio","video","three_d"]}
        elif not isinstance(self.modal, dict):
            self.modal = {}
        if not self.modal:
            default_vocab = getattr(self, "vocab_size", 128000)
            def_vocab_text = default_vocab if default_vocab else 128000
            self.modal = {
                "text": ModalConfig(vocab_size=def_vocab_text, modal_id=0),
                "image": ModalConfig(vocab_size=getattr(self, "image_vocab_size", 0) or 16384, modal_id=1),
                "audio": ModalConfig(vocab_size=getattr(self, "audio_vocab_size", 0) or 8192, modal_id=2),
                "video": ModalConfig(vocab_size=getattr(self, "video_vocab_size", 0) or 4096, modal_id=3),
                "three_d": ModalConfig(vocab_size=getattr(self, "three_d_vocab_size", 0) or 2048, modal_id=4),
            }
        for k in ["text","image","audio","video","three_d"]:
            if k not in self.modal:
                self.modal[k] = ModalConfig()
            elif isinstance(self.modal[k], dict):
                self.modal[k] = ModalConfig(**self.modal[k])
        if isinstance(self.heads, HeadsConfig):
            single = self.heads
            self.heads = {k: single for k in ["text","image","audio","video","three_d"]}
        elif not isinstance(self.heads, dict):
            self.heads = {}
        if not self.heads:
            default_vocab = getattr(self, "vocab_size", 128000)
            self.heads = {
                "text": HeadsConfig(vocab_size=default_vocab),
                "image": HeadsConfig(),
                "audio": HeadsConfig(),
                "video": HeadsConfig(),
                "three_d": HeadsConfig(),
            }
        for k in ["text","image","audio","video","three_d"]:
            if k not in self.heads:
                self.heads[k] = HeadsConfig()
            elif isinstance(self.heads[k], dict):
                self.heads[k] = HeadsConfig(**self.heads[k])
        return self

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        return self.hidden_size // self.num_attention_heads

    @property
    def hidden_dim(self) -> int:
        return self.hidden_size

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def ffn_dim(self) -> int:
        return self.intermediate_size

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    @property
    def eos_token_id(self) -> Optional[int]:
        if self.eos_token_id_ is not None:
            return self.eos_token_id_
        return 2

    @property
    def model_name(self) -> str:
        return self.name

def _flatten_yaml(data: Any, base_dir: Path) -> dict:
    if isinstance(data, dict) and "_base_" in data:
        bases = data.pop("_base_")
        merged: dict = {}
        for b in bases:
            bp = (base_dir / b).resolve()
            if bp.exists():
                import yaml
                try:
                    with open(bp, "r", encoding="utf-8") as f:
                        sub = yaml.safe_load(f) or {}
                except Exception:
                    sub = {}
                sub = _flatten_yaml(sub, bp.parent)
                merged = _deep_merge(merged, sub)
        merged = _deep_merge(merged, data)
        return merged
    return data if isinstance(data, dict) else {}

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config_from_yaml(path: str | Path) -> SkyModelConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _flatten_yaml(raw, p.parent)
    model_block = raw.get("model") if isinstance(raw, dict) else None
    if not isinstance(model_block, dict):
        raise ValueError(f"YAML must contain top-level 'model' dict key: {p}")
    modal_raw = model_block.get("modal") or {}
    modal_out: dict[str, ModalConfig] = {}
    for k in ("text", "image", "audio", "video", "three_d"):
        m = modal_raw.get(k) or {}
        modal_out[k] = ModalConfig(**m) if isinstance(m, dict) else ModalConfig()
    heads_raw = model_block.get("heads") or {}
    heads_out: dict[str, HeadsConfig] = {}
    for k in ("text", "image", "audio", "video", "three_d"):
        h = heads_raw.get(k) or {}
        heads_out[k] = HeadsConfig(**h) if isinstance(h, dict) else HeadsConfig()
    cfg_block = {**model_block, "modal": modal_out, "heads": heads_out}
    try:
        return SkyModelConfig(**cfg_block)
    except ValidationError as e:
        raise ValueError(f"Invalid model config at {p}: {e}") from e

def build_model_from_config(cfg: SkyModelConfig):
    from .sky_model import SkyModel
    return SkyModel(cfg)
