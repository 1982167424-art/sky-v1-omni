from pathlib import Path
import torch
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config
from sky_v1.training.checkpoint import CheckpointManager
from sky_v1.training.trainer import SkyTrainer

def _cfg(vocab_size, num_frames, num_points, mesh_vertices):
    heads = {k: HeadsConfig(
        vocab_size=vocab_size, num_frames=num_frames, num_points=num_points,
        point_dim=3, mesh_vertices=mesh_vertices, patch_size=16, mel_bins=128, out_channels=3
    ) for k in ["text","image","audio","video","three_d"]}
    modal = {k: ModalConfig(modal_id=i, image_size=64, frame_size=64, num_frames=num_frames, num_points=num_points, mesh_vertices=mesh_vertices, patch_size=16) for i,k in enumerate(["text","image","audio","video","three_d"])}
    return SkyModelConfig(
        name="mini", hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=160, vocab_size=vocab_size, max_position_embeddings=2048,
        rope_theta=10000, rms_norm_eps=1e-6, modal=modal, heads=heads,
    )

def test_ckpt_save_best_and_rollback(tmp_path):
    cfg = _cfg(128, 2, 16, 8)
    model = build_model_from_config(cfg)
    trainer = SkyTrainer(model, phase="phase1", learning_rate=1e-3, device="cpu", vocab_size=128)
    ckpt_dir = tmp_path / "ckpts"
    mgr = CheckpointManager(ckpt_dir, keep_last_k=3)
    step = 0
    for loss in [10.0, 5.0, 7.0, 3.0]:
        step += 1
        mgr.on_step_end(step=step, model=model, optimizer=trainer.optimizer, loss=loss)
    best = mgr.best_state()
    assert best is not None and best["loss"] == 3.0
