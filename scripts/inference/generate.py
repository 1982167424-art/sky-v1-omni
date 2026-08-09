"""Multi-modal generation runner.

Usage:
  python -m scripts.inference.generate --modality image --prompt "a cat" --steps 1
  python -m scripts.inference.generate --modality text  --prompt "hello"   --steps 2
  python -m scripts.inference.generate --modality audio --prompt "voice"   --steps 1
  python -m scripts.inference.generate --modality video --prompt "cat running" --steps 1
  python -m scripts.inference.generate --modality 3d    --prompt "chair"   --steps 1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("generate")
    p.add_argument("--modality", required=True, choices=["text", "image", "audio", "video", "3d"])
    p.add_argument("--prompt", required=True, type=str)
    p.add_argument("--steps", type=int, default=1, help="Number of decode steps (informational)")
    p.add_argument("--model", default="mini-sdk")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=str, default=None, help="Output path JSON")
    return p.parse_args()


def _tensor_meta(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return {"dtype": str(v.dtype), "shape": list(v.shape)}
    except Exception:
        pass
    if hasattr(v, "shape"):
        return {"shape": list(v.shape)}
    return None


def main() -> int:
    args = parse_args()
    from sky_v1.sdk.client import SkySDK
    sdk = SkySDK(engine="direct", model_name=args.model, device=args.device)
    out = sdk.generate(modality=args.modality, prompt=args.prompt, steps=args.steps)
    serializable: dict = {}
    for k, v in out.items():
        meta = _tensor_meta(v)
        serializable[k] = meta if meta is not None else v
    txt = json.dumps(serializable, indent=2, default=str)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
