"""Download public preference datasets for DPO (UltraFeedback / HH-RLHF).

Implements spec §11: 数据集下载失败 → 脚本提供 Toy Dataset + 公开数据集多镜像源.
On any failure the script falls back to toy preference pairs so DPO training
can always start. Output is normalized to ``{prompt, chosen, rejected}``.

Usage:
    python -m scripts.data.download_preference \\
        --output-dir ./data/preference --dataset ultrafeedback --source hf --max-samples 1000

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


DATASETS = ["ultrafeedback", "hh-rlhf"]
SOURCES = ["hf", "aliyun", "modelscope"]

# Per-dataset raw-key -> normalized-key map.
# hh-rlhf has no explicit "prompt" column: chosen/rejected share a common
# prefix of assistant+user turns which we extract as the prompt.
_KEY_MAP: dict[str, dict[str, str | None]] = {
    "ultrafeedback": {"prompt": "prompt",   "chosen": "chosen",   "rejected": "rejected"},
    "hh-rlhf":       {"prompt": None,       "chosen": "chosen",   "rejected": "rejected"},
}

# Toy preference pairs (prompt, chosen, rejected).
TOY_PAIRS: list[tuple[str, str, str]] = [
    ("What is the capital of France?", "The capital of France is Paris.", "I don't know."),
    ("Write a haiku about the sea.",
     "Blue waves softly crash,\nSalt upon the gentle breeze,\nLife beneath the tide.",
     "Sea is big."),
    ("Explain recursion.",
     "Recursion is a function that calls itself to solve smaller subproblems until a base case is reached.",
     "Recursion is when code repeats."),
    ("Translate 'good morning' to Spanish.", "Buenos días.", "Hola."),
    ("What is 7 * 8?", "56.", "54."),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.data.download_preference",
        description=(
            "下载偏好对数据集 (UltraFeedback / HH-RLHF)，格式 {prompt, chosen, rejected}，"
            "失败时生成 toy 偏好对。"
        ),
    )
    p.add_argument("--output-dir", type=str, default="./data/preference")
    p.add_argument("--source", choices=SOURCES, default="hf")
    p.add_argument("--dataset", choices=DATASETS, default="ultrafeedback")
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
    prompt_key = keys.get("prompt")
    chosen_key = keys.get("chosen") or "chosen"
    rejected_key = keys.get("rejected") or "rejected"

    if prompt_key is None:
        # hh-rlhf: chosen/rejected are message lists sharing a common prefix.
        chosen_list = sample.get(chosen_key) or []
        rejected_list = sample.get(rejected_key) or []
        if isinstance(chosen_list, list) and isinstance(rejected_list, list):
            n = min(len(chosen_list), len(rejected_list))
            common = 0
            for i in range(n):
                if chosen_list[i] == rejected_list[i]:
                    common = i + 1
                else:
                    break
            prompt = _extract_text(chosen_list[:common])
            chosen_text = _extract_text(chosen_list[common:])
            rejected_text = _extract_text(rejected_list[common:])
            return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}
        return {
            "prompt": "",
            "chosen": _extract_text(chosen_list),
            "rejected": _extract_text(rejected_list),
        }

    prompt = _extract_text(sample.get(prompt_key) or sample.get("prompt"))
    chosen_text = _extract_text(sample.get(chosen_key) or sample.get("chosen"))
    rejected_text = _extract_text(sample.get(rejected_key) or sample.get("rejected"))
    return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


def _toy_pairs(max_samples: int | None) -> list[dict[str, Any]]:
    n = max_samples if (max_samples is not None and max_samples > 0) else len(TOY_PAIRS)
    n = min(n, len(TOY_PAIRS))
    return [
        {"prompt": t[0], "chosen": t[1], "rejected": t[2], "toy": True}
        for t in TOY_PAIRS[:n]
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
    out_file = out_dir / f"{args.dataset}_preference.jsonl"

    print(
        f"[Preference] dataset={args.dataset} source={args.source} "
        f"max_samples={args.max_samples}"
    )
    print(f"[Preference] target: {out_file}")

    loader = RealDatasetLoader()
    raw = loader.load(
        args.dataset,
        split="train",
        source=args.source,
        max_samples=args.max_samples,
    )

    if raw and all(isinstance(r, dict) and r.get("toy") for r in raw):
        samples = _toy_pairs(args.max_samples)
    else:
        samples = [_normalize(r, args.dataset) for r in raw if isinstance(r, dict)]
        if not samples:
            samples = _toy_pairs(args.max_samples)

    n_written = _to_jsonl(samples, out_file)
    n_toy = sum(1 for s in samples if isinstance(s, dict) and s.get("toy"))

    print(f"[Preference] written: {n_written} samples -> {out_file}")
    print(f"[Preference] toy samples: {n_toy} / {n_written}")
    if n_toy == n_written:
        status = "FALLBACK_TOY"
    elif n_toy == 0:
        status = "REAL_DATA"
    else:
        status = "MIXED"
    print(f"[Preference] status: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
