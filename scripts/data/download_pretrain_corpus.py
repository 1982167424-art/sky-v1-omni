"""Download public pretraining corpora (RedPajama / Wikipedia / OpenWebText).

Implements spec §11: 数据集下载失败 → 脚本提供 Toy Dataset + 公开数据集多镜像源.
When the requested public dataset cannot be fetched, the script falls back to
deterministic toy samples so downstream training can always start.

Usage:
    python -m scripts.data.download_pretrain_corpus \\
        --output-dir ./data/pretrain --source hf --dataset redpajama --max-samples 1000

Sources (mirrors):
    hf        : HuggingFace datasets hub (huggingface.co)
    aliyun    : Aliyun Tianchi HF mirror (hf-mirror.com)
    modelscope: ModelScope 魔搭社区 (modelscope.cn)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sky_v1.data.real_datasets import RealDatasetLoader


DATASETS = ["redpajama", "wikipedia", "openwebtext"]
SOURCES = ["hf", "aliyun", "modelscope"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.data.download_pretrain_corpus",
        description=(
            "下载公开预训练语料 (RedPajama / Wikipedia / OpenWebText)，"
            "失败时 fallback 到 toy 样本。"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./data/pretrain",
        help="输出目录 (默认: ./data/pretrain)",
    )
    p.add_argument(
        "--source",
        choices=SOURCES,
        default="hf",
        help="数据源镜像 (默认: hf)",
    )
    p.add_argument(
        "--dataset",
        choices=DATASETS,
        default="redpajama",
        help="数据集名 (默认: redpajama)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数 (默认: 全部 / toy 默认 16)",
    )
    return p


def _to_jsonl(samples: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    args = _build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}.jsonl"

    print(
        f"[PretrainCorpus] dataset={args.dataset} source={args.source} "
        f"max_samples={args.max_samples}"
    )
    print(f"[PretrainCorpus] target: {out_file}")

    loader = RealDatasetLoader()
    samples = loader.load(
        args.dataset,
        split="train",
        source=args.source,
        max_samples=args.max_samples,
    )

    n_written = _to_jsonl(samples, out_file)
    n_toy = sum(1 for s in samples if isinstance(s, dict) and s.get("toy"))

    print(f"[PretrainCorpus] written: {n_written} samples -> {out_file}")
    print(f"[PretrainCorpus] toy samples: {n_toy} / {n_written}")
    if n_toy == n_written:
        status = "FALLBACK_TOY"
    elif n_toy == 0:
        status = "REAL_DATA"
    else:
        status = "MIXED"
    print(f"[PretrainCorpus] status: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
