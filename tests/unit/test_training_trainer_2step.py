import torch
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config
from sky_v1.training.trainer import SkyTrainer
from sky_v1.data.toy_generator import ToyDataGenerator
from sky_v1.data.datasets import Phase3DistillDataset
from sky_v1.data.collator import SkyDataCollator
from torch.utils.data import DataLoader

def _cfg(vocab_size, num_frames, num_points, mesh_vertices):
    heads = {k: HeadsConfig(
        vocab_size=vocab_size, num_frames=num_frames, num_points=num_points,
        point_dim=3, mesh_vertices=mesh_vertices, patch_size=16, mel_bins=128, out_channels=3
    ) for k in ["text","image","audio","video","three_d"]}
    modal = {k: ModalConfig(modal_id=i, image_size=32, frame_size=32, num_frames=num_frames, num_points=num_points, mesh_vertices=mesh_vertices, patch_size=16) for i,k in enumerate(["text","image","audio","video","three_d"])}
    return SkyModelConfig(
        name="mini", hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=160, vocab_size=vocab_size, max_position_embeddings=4096,
        rope_theta=10000, rms_norm_eps=1e-6, modal=modal, heads=heads,
    )

def test_trainer_phase3_2step_loss_decreases():
    vocab_size = 500
    cfg = _cfg(vocab_size, 1, 8, 4)
    torch.manual_seed(0)
    model = build_model_from_config(cfg)
    trainer = SkyTrainer(model, phase="phase3", learning_rate=3e-3, weight_decay=0.0, device="cpu", vocab_size=vocab_size)
    samples = list(ToyDataGenerator(n=8, seed=1, vocab_size=vocab_size, text_len=8, image_size=32, audio_frames=8, video_frames=1, three_d_points=8, three_d_mesh_verts=4).generate_all())
    ds = Phase3DistillDataset(samples, vocab_size=vocab_size)
    loader = DataLoader(ds, batch_size=4, collate_fn=SkyDataCollator(max_seq_len=32), shuffle=True)
    losses = []
    for epoch in range(5):
        epoch_losses = []
        for batch in loader:
            loss, _ = trainer.step(batch)
            epoch_losses.append(float(loss))
        losses.append(sum(epoch_losses) / len(epoch_losses))
    assert len(losses) >= 2
    assert losses[-1] <= losses[0] + 1e-2, f"epoch avg loss must decrease or be stable: {losses}"
