import torch
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config
from sky_v1.training.trainer import SkyTrainer
from sky_v1.data.toy_generator import ToyDataGenerator
from sky_v1.data.datasets import Phase3DistillDataset
from sky_v1.data.collator import SkyDataCollator
from torch.utils.data import DataLoader

def _cfg(vocab_size: int = 500):
    heads = {k: HeadsConfig(
        vocab_size=vocab_size, num_frames=2, num_points=16, point_dim=3,
        mesh_vertices=8, patch_size=16, mel_bins=128, out_channels=3,
    ) for k in ["text","image","audio","video","three_d"]}
    modal = {k: ModalConfig(
        modal_id=i, image_size=64, frame_size=64, num_frames=2,
        num_points=16, mesh_vertices=8, patch_size=16,
    ) for i,k in enumerate(["text","image","audio","video","three_d"])}
    return SkyModelConfig(
        name="smoke", hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=160, vocab_size=vocab_size, max_position_embeddings=512,
        rope_theta=10000, rms_norm_eps=1e-6, modal=modal, heads=heads,
    )

def test_m2m3_smoke_phase3_no_crash():
    vocab_size = 500
    torch.manual_seed(1)
    cfg = _cfg(vocab_size)
    model = build_model_from_config(cfg)
    trainer = SkyTrainer(model, phase="phase3", learning_rate=1e-3, weight_decay=0.0, device="cpu", vocab_size=vocab_size)
    samples = list(ToyDataGenerator(n=3, seed=42, vocab_size=vocab_size).generate_all())
    ds = Phase3DistillDataset(samples, vocab_size=vocab_size)
    loader = DataLoader(ds, batch_size=1, collate_fn=SkyDataCollator(max_seq_len=64))
    losses = []
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        loss, metrics = trainer.step(batch)
        lv = float(loss)
        assert torch.isfinite(torch.tensor(lv)), f"Non-finite loss at step {i}: {lv}"
        losses.append(lv)
    assert len(losses) == 2
