"""Interactive chat CLI using SkyInferenceEngine or remote HTTP SDK.

Usage:
  python -m scripts.inference.chat --engine direct --model mini-sdk
  python -m scripts.inference.chat --engine http --base-url http://localhost:8000/v1
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("chat")
    p.add_argument("--engine", choices=["direct", "http"], default="direct")
    p.add_argument("--model", default="mini-sdk")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--message", "-m", type=str, default=None, help="Single-shot message")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from sky_v1.sdk.client import SkySDK
    sdk = SkySDK(engine=args.engine, base_url=args.base_url, model_name=args.model)
    history: list[dict] = []
    if args.message:
        history.append({"role": "user", "content": args.message})
        resp = sdk.chat_completions(history, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("Assistant:", resp["choices"][0]["message"]["content"])
        return 0
    print("sky-v1 chat (type 'exit' to quit)")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user.lower() in {"exit", "quit", "q"}:
            return 0
        if not user:
            continue
        history.append({"role": "user", "content": user})
        resp = sdk.chat_completions(history, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        txt = resp["choices"][0]["message"]["content"]
        print("Assistant:", txt)
        history.append({"role": "assistant", "content": txt})


if __name__ == "__main__":
    sys.exit(main())
