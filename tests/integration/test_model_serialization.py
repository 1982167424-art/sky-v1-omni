import torch
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config

def _make_modal_heads_cfg(vocab_size, num_frames, num_points, mesh_vertices):
    return {k: HeadsConfig(
        vocab_size=vocab_size, num_frames=num_frames, num_points=num_points,
        point_dim=3, mesh_vertices=mesh_vertices, patch_size=16, mel_bins=128, out_channels=3
    ) for k in ["text","image","audio","video","three_d"]}

def _make_mini():
    return SkyModelConfig(
        name="mini", hidden_size=96, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=256, vocab_size=500, max_position_embeddings=512,
        rope_theta=10000, rms_norm_eps=1e-6,
        modal={k: ModalConfig(modal_id=i, image_size=64, frame_size=64, num_frames=2, num_points=32, mesh_vertices=16, patch_size=16) for i,k in enumerate(["text","image","audio","video","three_d"])},
        heads=_make_modal_heads_cfg(500, 2, 32, 16),
    )

def test_save_load_forward_matches(tmp_path):
    cfg = _make_mini()
    m1 = build_model_from_config(cfg).eval()
    inp = {
        "text": torch.randint(0, 500, (1, 16)),
        "image": torch.randn(1, 3, 64, 64),
        "audio": torch.randn(1, 128, 16),
        "video": torch.randn(1, 2, 3, 64, 64),
        "three_d": (torch.randn(1, 32, 6), torch.randn(1, 16, 3)),
    }
    with torch.no_grad():
        out1 = m1(inp)
    path = tmp_path / "sky_min.pt"
    torch.save({"state_dict": m1.state_dict(), "config": cfg.model_dump(mode="json")}, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg2 = SkyModelConfig(**ckpt["config"])
    m2 = build_model_from_config(cfg2).eval()
    m2.load_state_dict(ckpt["state_dict"])
    with torch.no_grad():
        out2 = m2(inp)
    assert torch.allclose(out1["text"], out2["text"], atol=1e-5)
    assert torch.allclose(out1["image"], out2["image"], atol=1e-5)
