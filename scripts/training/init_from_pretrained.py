"""Warm-start a SkyModel from SOTA pretrained modal tokenizers (spec §2.2 Scheme B).

Builds a SkyModel from a YAML config, attempts to load pretrained weights
into each modality tokenizer, then saves the resulting checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sky_v1.model.config import load_config_from_yaml, build_model_from_config
from sky_v1.model.pretrained_loader import load_all_pretrained


def _build_pretrained_overrides(cfg, text_repo, image_repo, audio_repo) -> dict:
    """Merge config's ``pretrained`` block with explicit CLI flags (CLI wins)."""
    base: dict = {}
    if hasattr(cfg, "pretrained") and isinstance(cfg.pretrained, dict):
        base = dict(cfg.pretrained)
    if text_repo:
        base["text"] = text_repo
    if image_repo:
        base["image"] = image_repo
    if audio_repo:
        base["audio"] = audio_repo
    return base


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Warm-start a SkyModel from pretrained modal tokenizers"
    )
    ap.add_argument("--model-config", required=True, help="Path to model YAML config")
    ap.add_argument("--output", required=True, help="Output checkpoint path (.pt)")
    ap.add_argument("--text-repo", default=None, help="HuggingFace Qwen repo or local path")
    ap.add_argument("--image-repo", default=None, help="HuggingFace CLIP ViT repo or local path")
    ap.add_argument("--audio-repo", default=None, help="HuggingFace Whisper repo or local path")
    args = ap.parse_args()

    cfg_path = ROOT / args.model_config if not Path(args.model_config).is_absolute() else Path(args.model_config)
    mcfg = load_config_from_yaml(cfg_path)
    model = build_model_from_config(mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init_from_pretrained] model params: {n_params:,}")

    pretrained = _build_pretrained_overrides(mcfg, args.text_repo, args.image_repo, args.audio_repo)
    report = load_all_pretrained(model, {"pretrained": pretrained})

    print("[init_from_pretrained] pretrained warm-start report:")
    for modal in ("text", "image", "audio"):
        status = report.get(modal, "skipped")
        marker = {"loaded": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(status, "?")
        print(f"  [{marker}] {modal:<6} -> {status}")

    out_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "model_state_dict": model.state_dict(),
        "config_name": mcfg.name,
        "pretrained_report": report,
    }
    torch.save(save, out_path)
    print(f"[init_from_pretrained] saved warm-start checkpoint -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
