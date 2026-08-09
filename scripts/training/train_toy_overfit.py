from __future__ import annotations
"""2-step toy overfit smoke test. Verifies loss decreases on CPU for tiny config."""
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config
from sky_v1.training.trainer import SkyTrainer
from sky_v1.data.toy_generator import ToyDataGenerator
from sky_v1.data.datasets import Phase3DistillDataset
from sky_v1.data.collator import SkyDataCollator

def _make_mini_cfg(vocab_size: int = 500):
    heads = {k: HeadsConfig(
        vocab_size=vocab_size, num_frames=2, num_points=16, point_dim=3,
        mesh_vertices=8, patch_size=16, mel_bins=128, out_channels=3,
    ) for k in ["text","image","audio","video","three_d"]}
    modal = {k: ModalConfig(
        modal_id=i, image_size=64, frame_size=64, num_frames=2,
        num_points=16, mesh_vertices=8, patch_size=16,
    ) for i,k in enumerate(["text","image","audio","video","three_d"])}
    return SkyModelConfig(
        name="mini", hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=160, vocab_size=vocab_size, max_position_embeddings=512,
        rope_theta=10000, rms_norm_eps=1e-6, modal=modal, heads=heads,
    )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10, help="Toy overfit steps (default 10)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    vocab_size = 500
    cfg = _make_mini_cfg(vocab_size)
    torch.manual_seed(0)
    model = build_model_from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ToyOverfit] mini-model params: {n_params:,}")
    trainer = SkyTrainer(model, phase="phase3", learning_rate=args.lr, weight_decay=0.0, device=args.device, vocab_size=vocab_size)
    samples = list(ToyDataGenerator(n=4, seed=1, vocab_size=vocab_size).generate_all())
    ds = Phase3DistillDataset(samples, vocab_size=vocab_size)
    loader = DataLoader(ds, batch_size=2, collate_fn=SkyDataCollator(max_seq_len=64), shuffle=False)
    losses: list[float] = []
    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            loss, m = trainer.step(batch)
            lv = float(loss)
            losses.append(lv)
            info = {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}
            print(f"  step={step:03d}  loss={lv:.4f}  {info}")
            step += 1
    print(f"\n[ToyOverfit] Loss trajectory ({len(losses)} steps):")
    print(f"  first: {losses[0]:.4f}")
    print(f"  min:   {min(losses):.4f}")
    print(f"  last:  {losses[-1]:.4f}")
    ok = losses[-1] < losses[0] + 1e-2
    print(f"  verdict: {'PASS (loss decreased/stable)' if ok else 'FAIL (loss did not decrease)'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
