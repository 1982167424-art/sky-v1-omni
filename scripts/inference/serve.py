"""Launch inference/API server (FastAPI + Uvicorn).

Usage:
  # Agent 模式（默认）：model=sky-v1-agent 走 SkyAgent
  python -m scripts.inference.serve --host 0.0.0.0 --port 8000

  # Engine 模式：model=sky-v1-* 走 SkyInferenceEngine 真实推理
  python -m scripts.inference.serve --engine --port 8000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=str, default=None, help="inference config YAML")
    p.add_argument("--model-config", type=str, default=None, help="model config YAML")
    p.add_argument("--engine", action="store_true",
                   help="启用 SkyInferenceEngine（model=sky-v1-* 走真实模型推理）")
    p.add_argument("--reload", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    import uvicorn
    if args.engine:
        # Engine 模式：构造带引擎的 app 实例（reload 需字符串导入路径，与实例不兼容，故禁用）
        from sky_v1.api.app import create_app
        app = create_app(enable_engine=True)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Agent 模式（默认）：保持工厂导入，支持 reload
        uvicorn.run(
            "sky_v1.api.app:create_app",
            host=args.host, port=args.port,
            reload=args.reload, factory=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
