import torch
from sky_v1.model.transformer_layer import UniTransformerLayer
from sky_v1.model.config import SkyModelConfig

def test_transformer_layer_residual_shape_and_no_nan():
    cfg = SkyModelConfig(
        hidden_size=256, num_hidden_layers=1, num_attention_heads=8,
        intermediate_size=640, vocab_size=1000, max_position_embeddings=512,
        rope_theta=10000, rms_norm_eps=1e-6,
    )
    layer = UniTransformerLayer(cfg, layer_idx=0)
    b, s = 2, 17
    x = torch.randn(b, s, cfg.hidden_size)
    out = layer(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    out.sum().backward()
