"""Launch inference/API server (FastAPI + Uvicorn).

Usage:
  python -m scripts.inference.serve --host 0.0.0.0 --port 8000
  python -m scripts.inference.serve --config configs/inference/sky_v1_infer_cpu.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=str, default=None, help="inference config YAML")
    p.add_argument("--model-config", type=str, default=None, help="model config YAML")
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn
    uvicorn.run(
        "sky_v1.api.app:create_app",
        host=args.host, port=args.port,
        reload=args.reload, factory=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
