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
from sky_v1.data.datasets import Phase3DistillDataset
from sky_v1.data.collator import SkyDataCollator

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/sky_v1_1B.yaml")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-dir", default="outputs/phase3_smoke")
    args = ap.parse_args()

    mcfg = load_config_from_yaml(ROOT / args.config)
    model = build_model_from_config(mcfg)
    print(f"[Phase3] params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = SkyTrainer(model, phase="phase3", learning_rate=5e-5, device=args.device, vocab_size=mcfg.vocab_size)
    samples = list(ToyDataGenerator(n=8, seed=42, vocab_size=mcfg.vocab_size).generate_all())
    ds = Phase3DistillDataset(samples, vocab_size=mcfg.vocab_size, num_teachers=5)
    loader = DataLoader(ds, batch_size=2, collate_fn=SkyDataCollator(max_seq_len=256))
    ckpt = CheckpointManager(ROOT / args.output_dir, keep_last_k=3)
    logger = MetricsLogger(ROOT / args.output_dir / "logs")
    step = 0
    losses = []
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            loss, m = trainer.step(batch)
            losses.append(float(loss))
            logger.log(step=step, loss=float(loss), **m)
            print(f"[Phase3] step={step} loss={float(loss):.4f} metrics={ {k:v for k,v in m.items() if isinstance(v, (int,float))} }")
            ckpt.on_step_end(step, model, trainer.optimizer, float(loss))
            step += 1
    logger.close()
    if len(losses) >= 2:
        print(f"[Phase3] loss progression: {losses}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
