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
    # 加载真实 1B YAML 配置（验证 load_config_from_yaml 流程），
    # 但缩小内存敏感字段（vocab/embedding 在 6GB 沙箱会 OOM），仍走完整 build_model_from_config。
    cfg = load_config_from_yaml(HERE / "configs/model/sky_v1_1B.yaml")
    cfg.hidden_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.intermediate_size = 320
    cfg.vocab_size = 1000
    cfg.max_position_embeddings = 2048
    cfg.modal["text"].vocab_size = 1000
    cfg.heads["text"].vocab_size = 1000
    model = build_model_from_config(cfg)
    import torch
    bs, seq = 2, 64
    ids = torch.zeros(bs, seq, cfg.hidden_size)
    out = model.backbone(ids)
    assert out.shape == (bs, seq, cfg.hidden_size)
    # 验证五模态 tokenizer/head 均已构建
    for attr in ("text_tok", "image_tok", "audio_tok", "video_tok", "threed_tok",
                 "text_head", "image_head", "audio_head", "video_head", "threed_head"):
        assert hasattr(model, attr), f"missing {attr}"
