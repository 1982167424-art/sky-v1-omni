"""Download public SFT instruction datasets (Alpaca / Dolly / Orca / UltraChat).

Implements spec §11: 数据集下载失败 → 脚本提供 Toy Dataset + 公开数据集多镜像源.
On any failure the script falls back to 10 toy SFT templates so training can
always start. Output is normalized to ``{instruction, input, output}``.

Usage:
    python -m scripts.data.download_sft_dataset \\
        --output-dir ./data/sft --dataset alpaca --source hf --max-samples 1000

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
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sky_v1.data.real_datasets import RealDatasetLoader


DATASETS = ["alpaca", "dolly", "orca", "ultrachat"]
SOURCES = ["hf", "aliyun", "modelscope"]

# Per-dataset raw-key -> normalized-key map. Different public SFT datasets
# expose different column names; this lets us converge on {instruction,input,output}.
_KEY_MAP: dict[str, dict[str, str | None]] = {
    "alpaca":    {"instruction": "instruction", "input": "input",    "output": "output"},
    "dolly":     {"instruction": "instruction", "input": "context", "output": "response"},
    "orca":      {"instruction": "question",    "input": None,       "output": "answer"},
    "ultrachat": {"instruction": "messages",    "input": None,       "output": None},
}

# 10 hand-written toy SFT samples (instruction, input, output).
TOY_SFT_TEMPLATES: list[tuple[str, str, str]] = [
    ("请用一句话解释机器学习。", "", "机器学习是让计算机从数据中自动学习规律并做出预测或决策的方法。"),
    ("翻译成英文：今天天气真好。", "", "The weather is nice today."),
    ("写一个 Python 函数计算阶乘。", "输入: 5", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"),
    ("总结这段话的要点。", "GPT 是基于 Transformer 的语言模型。", "要点：1) 基于 Transformer；2) 是一种语言模型。"),
    ("写一首关于秋天的诗。", "", "秋风起处叶纷飞，\n一地金黄映日辉。"),
    ("解释什么是闭包。", "", "闭包是函数与其引用的外部变量的组合，使得函数能访问外部作用域的变量。"),
    ("把这句话改写得更正式。", "这玩意儿挺好用的。", "该产品具有良好的实用性能。"),
    ("列出三种排序算法。", "", "1) 冒泡排序 2) 快速排序 3) 归并排序"),
    ("将下面的数字按升序排列。", "5 2 8 1 9", "1 2 5 8 9"),
    ("写一封简短的请假邮件。", "请假一天", "尊敬的领导：因个人原因，本人申请明日请假一天，恳请批准。"),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.data.download_sft_dataset",
        description=(
            "下载公开 SFT 指令数据集 (Alpaca / Dolly / Orca / UltraChat)，"
            "统一格式 {instruction, input, output}，失败时生成 10 条 toy SFT 样本。"
        ),
    )
    p.add_argument("--output-dir", type=str, default="./data/sft")
    p.add_argument("--source", choices=SOURCES, default="hf")
    p.add_argument("--dataset", choices=DATASETS, default="alpaca")
    p.add_argument("--max-samples", type=int, default=None)
    return p


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for m in value:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(value)


def _normalize(sample: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    keys = _KEY_MAP.get(dataset_name, {})
    inst_key = keys.get("instruction") or "instruction"
    in_key = keys.get("input")
    out_key = keys.get("output") or "output"

    instruction = _extract_text(sample.get(inst_key) or sample.get("instruction"))
    output = _extract_text(sample.get(out_key) or sample.get("output"))
    if in_key:
        inp = _extract_text(sample.get(in_key))
    else:
        inp = _extract_text(sample.get("input"))
    return {"instruction": instruction, "input": inp, "output": output}


def _toy_sft_samples(max_samples: int | None) -> list[dict[str, Any]]:
    n = max_samples if (max_samples is not None and max_samples > 0) else len(TOY_SFT_TEMPLATES)
    n = min(n, len(TOY_SFT_TEMPLATES))
    return [
        {"instruction": t[0], "input": t[1], "output": t[2], "toy": True}
        for t in TOY_SFT_TEMPLATES[:n]
    ]


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
    out_file = out_dir / f"{args.dataset}_sft.jsonl"

    print(
        f"[SFT] dataset={args.dataset} source={args.source} "
        f"max_samples={args.max_samples}"
    )
    print(f"[SFT] target: {out_file}")

    loader = RealDatasetLoader()
    raw = loader.load(
        args.dataset,
        split="train",
        source=args.source,
        max_samples=args.max_samples,
    )

    # If RealDatasetLoader already fell back to toy, replace with the richer
    # hand-written SFT templates (better instruction-following signal).
    if raw and all(isinstance(r, dict) and r.get("toy") for r in raw):
        samples = _toy_sft_samples(args.max_samples)
    else:
        samples = [_normalize(r, args.dataset) for r in raw if isinstance(r, dict)]
        if not samples:
            samples = _toy_sft_samples(args.max_samples)

    n_written = _to_jsonl(samples, out_file)
    n_toy = sum(1 for s in samples if isinstance(s, dict) and s.get("toy"))

    print(f"[SFT] written: {n_written} samples -> {out_file}")
    print(f"[SFT] toy samples: {n_toy} / {n_written}")
    if n_toy == n_written:
        status = "FALLBACK_TOY"
    elif n_toy == 0:
        status = "REAL_DATA"
    else:
        status = "MIXED"
    print(f"[SFT] status: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
