"""Download public multimodal datasets across image / audio / video / 3d.

Implements spec §11: 数据集下载失败 → 脚本提供 Toy Dataset + 公开数据集多镜像源.
On any failure the script generates toy multimodal samples (random tensors +
fake labels) so training pipelines can always start.

Recommended datasets per modality:
    image: COCO Captions / LAION-400M (sample)
    audio: LibriSpeech / AudioSet
    video: MSR-VTT / WebVid
    3d   : ShapeNet / Objaverse

Usage:
    python -m scripts.data.download_modal_datasets \\
        --output-dir ./data/modal --modality image --dataset coco_captions --max-samples 50
    python -m scripts.data.download_modal_datasets --modality all
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


MODALITIES = ["image", "audio", "video", "3d", "all"]

_DATASETS_BY_MODALITY: dict[str, list[str]] = {
    "image": ["coco_captions", "laion_400m_sample"],
    "audio": ["librispeech", "audioset"],
    "video": ["msr_vtt", "webvid"],
    "3d":    ["shapenet", "objaverse"],
}

# default toy tensor shape per modality (matches sky_v1.data.toy_generator sizing)
_TOY_SHAPES: dict[str, tuple[int, ...]] = {
    "image": (3, 64, 64),
    "audio": (128, 16),
    "video": (2, 3, 64, 64),
    "3d":    (32, 6),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.data.download_modal_datasets",
        description=(
            "下载多模态数据集 (image/audio/video/3d)，失败时生成 toy 多模态样本"
            "（随机张量 + 假标签）。"
        ),
    )
    p.add_argument("--output-dir", type=str, default="./data/modal")
    p.add_argument("--modality", choices=MODALITIES, default="all")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="具体数据集名 (默认: 各模态首个推荐)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    return p


def _toy_modal_sample(modality: str, idx: int) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(2024 * 7 + idx)
    shape = _TOY_SHAPES.get(modality, (8,))
    arr = rng.normal(size=shape).astype("float32")
    label = f"toy_{modality}_label_{idx}"
    return {
        "id": f"toy_{modality}_{idx}",
        "modality": modality,
        "tensor_shape": list(arr.shape),
        "label": label,
        "tensor": arr.tolist(),
        "toy": True,
    }


def _toy_modal_samples(modality: str, max_samples: int | None) -> list[dict[str, Any]]:
    n = max_samples if (max_samples is not None and max_samples > 0) else 8
    return [_toy_modal_sample(modality, i) for i in range(n)]


def _to_jsonl(samples: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def _run_one(
    modality: str,
    dataset: str,
    max_samples: int | None,
    out_dir: Path,
) -> dict[str, Any]:
    print(
        f"[Modal] modality={modality} dataset={dataset} "
        f"max_samples={max_samples}"
    )
    loader = RealDatasetLoader()
    raw = loader.load(
        dataset,
        split="train",
        source="hf",
        max_samples=max_samples,
    )

    if raw and all(isinstance(r, dict) and r.get("toy") for r in raw):
        samples = _toy_modal_samples(modality, max_samples)
    else:
        samples = [
            {"id": r.get("id", f"r{i}"), "modality": modality, "raw": r}
            for i, r in enumerate(raw)
            if isinstance(r, dict)
        ]
        if not samples:
            samples = _toy_modal_samples(modality, max_samples)

    out_file = out_dir / f"{modality}_{dataset}.jsonl"
    n_written = _to_jsonl(samples, out_file)
    n_toy = sum(1 for s in samples if isinstance(s, dict) and s.get("toy"))

    print(f"[Modal] written: {n_written} samples -> {out_file}")
    print(f"[Modal] toy samples: {n_toy} / {n_written}")
    return {
        "modality": modality,
        "dataset": dataset,
        "n_written": n_written,
        "n_toy": n_toy,
        "file": str(out_file),
    }


def main() -> int:
    args = _build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.modality == "all":
        targets: list[tuple[str, str]] = []
        for m in ["image", "audio", "video", "3d"]:
            ds = args.dataset if args.dataset else _DATASETS_BY_MODALITY[m][0]
            targets.append((m, ds))
    else:
        ds = args.dataset if args.dataset else _DATASETS_BY_MODALITY[args.modality][0]
        targets = [(args.modality, ds)]

    results: list[dict[str, Any]] = []
    for m, ds in targets:
        results.append(_run_one(m, ds, args.max_samples, out_dir))

    print("[Modal] summary:")
    for r in results:
        print(
            f"  {r['modality']:6s} {r['dataset']:24s} -> "
            f"{r['n_written']:5d} samples (toy={r['n_toy']})  {r['file']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
