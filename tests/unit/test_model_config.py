from pathlib import Path
import pytest
from sky_v1.model.config import SkyModelConfig, load_config_from_yaml, build_model_from_config

HERE = Path(__file__).resolve().parents[2]

def test_1b_config_fields_present():
    cfg = load_config_from_yaml(HERE / "configs/model/sky_v1_1B.yaml")
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 16
    assert cfg.intermediate_size == 5460
    assert cfg.vocab_size == 128000
    assert cfg.max_position_embeddings == 8192
    assert cfg.modal_types == ["text", "image", "audio", "video", "three_d"]
    assert 0.0 <= cfg.attention_dropout < 1.0

def test_build_model_from_config_shape():
    cfg = load_config_from_yaml(HERE / "configs/model/sky_v1_1B.yaml")
    model = build_model_from_config(cfg)
    import torch
    bs, seq = 2, 64
    ids = torch.zeros(bs, seq, cfg.hidden_size)
    out = model.backbone(ids)
    assert out.shape == (bs, seq, cfg.hidden_size)
