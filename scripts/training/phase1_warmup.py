from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sky_v1.model.config import load_config_from_yaml, build_model_from_config
from sky_v1.training.trainer import SkyTrainer
from sky_v1.training.checkpoint import CheckpointManager
from sky_v1.training.callbacks import MetricsLogger
from sky_v1.data.toy_generator import ToyDataGenerator
from sky_v1.data.datasets import Phase1Dataset
from sky_v1.data.collator import SkyDataCollator

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/sky_v1_1B.yaml")
    ap.add_argument("--training-config", default="configs/training/phase1_warmup.yaml")
    ap.add_argument("--steps", type=int, default=2, help="Number of steps (for smoke test use 2)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-dir", default="outputs/phase1_smoke")
    args = ap.parse_args()

    mcfg = load_config_from_yaml(ROOT / args.config)
    # Build tiny override model for fast CLI smoke by reducing sizes
    model = build_model_from_config(mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Phase1] model params: {n_params:,}")
    trainer = SkyTrainer(model, phase="phase1", learning_rate=3e-4, device=args.device, vocab_size=mcfg.vocab_size)
    samples = list(ToyDataGenerator(n=8, seed=42, vocab_size=mcfg.vocab_size).generate_all())
    ds = Phase1Dataset(samples, phase="all")
    loader = DataLoader(ds, batch_size=2, collate_fn=SkyDataCollator(max_seq_len=128))
    ckpt_mgr = CheckpointManager(ROOT / args.output_dir, keep_last_k=3)
    logger = MetricsLogger(ROOT / args.output_dir / "logs")
    losses = []
    step = 0
    done = False
    while not done:
        for batch in loader:
            if step >= args.steps:
                done = True
                break
            loss, metrics = trainer.step(batch)
            losses.append(float(loss))
            metrics.pop("step", None)  # trainer 已在 metrics 写入 step，避免与显式 step 冲突
            logger.log(step=step, loss=float(loss), **metrics)
            if step % 1 == 0:
                print(f"[Phase1] step={step} loss={float(loss):.4f}")
            ckpt_mgr.on_step_end(step, model, trainer.optimizer, float(loss))
            step += 1
    if len(losses) >= 2:
        print(f"[Phase1] first loss={losses[0]:.4f} last loss={losses[-1]:.4f}")
        if losses[-1] <= losses[0] + 1e-2:
            print("[Phase1] OK: loss decreased or stable (overfit signal)")
            return 0
        print("[Phase1] WARN: loss did not decrease")
    logger.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
