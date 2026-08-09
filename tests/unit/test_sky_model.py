import torch
from pathlib import Path
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config

def _make_modal_heads_cfg(vocab_size, num_frames, num_points, mesh_vertices):
    return {k: HeadsConfig(
        vocab_size=vocab_size, num_frames=num_frames, num_points=num_points,
        point_dim=3, mesh_vertices=mesh_vertices, patch_size=16, mel_bins=128, out_channels=3
    ) for k in ["text","image","audio","video","three_d"]}

def test_sky_model_forward_five_modal_inputs_shape():
    mini = SkyModelConfig(
        name="mini", hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=320, vocab_size=1000, max_position_embeddings=2048,
        rope_theta=10000, rms_norm_eps=1e-6,
        modal={k: ModalConfig(modal_id=i, image_size=64, frame_size=64, num_frames=4, num_points=64, mesh_vertices=16, patch_size=16) for i,k in enumerate(["text","image","audio","video","three_d"])},
        heads=_make_modal_heads_cfg(1000, 4, 64, 16),
    )
    m = build_model_from_config(mini)
    B = 2
    inputs = {
        "text": torch.randint(0, 1000, (B, 32)),
        "image": torch.randn(B, 3, 64, 64),
        "audio": torch.randn(B, 128, 40),
        "video": torch.randn(B, 4, 3, 64, 64),
        "three_d": (torch.randn(B, 64, 6), torch.randn(B, 16, 3)),
    }
    out = m(inputs)
    assert set(["text", "image", "audio", "video", "three_d"]).issubset(set(out.keys()))
    assert out["text"].shape[0] == B and out["text"].shape[-1] == 1000
    assert tuple(out["image"].shape) == (B, 3, 64, 64)
    assert tuple(out["video"].shape) == (B, 4, 3, 64, 64)
    pts, mv = out["three_d"]
    assert pts.shape == (B, 64, 3)
    assert mv.shape == (B, 16, 3)

def test_sky_model_backward_has_gradients():
    mini = SkyModelConfig(
        name="mini", hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=160, vocab_size=500, max_position_embeddings=1024,
        rope_theta=10000, rms_norm_eps=1e-6,
        modal={k: ModalConfig(modal_id=i, image_size=64, frame_size=64, num_frames=2, num_points=16, mesh_vertices=8, patch_size=16) for i,k in enumerate(["text","image","audio","video","three_d"])},
        heads=_make_modal_heads_cfg(500, 2, 16, 8),
    )
    m = build_model_from_config(mini)
    inputs = {
        "text": torch.randint(0, 500, (1, 8)),
        "image": torch.randn(1, 3, 64, 64),
        "audio": torch.randn(1, 128, 16),
        "video": torch.randn(1, 2, 3, 64, 64),
        "three_d": (torch.randn(1, 16, 6), torch.randn(1, 8, 3)),
    }
    out = m(inputs)
    loss = out["text"].sum() + out["image"].sum() + out["three_d"][0].sum()
    loss.backward()
    ok = any(p.grad is not None and (p.grad != 0).any() for p in m.parameters())
    assert ok, "No nonzero gradients after backward!"
