"""构建 5 老师蒸馏数据集脚本（规格 §6 / §11）。

调用 :class:`sky_v1.training.teacher_client.DistillSetBuilder` 批量调 5 个老师模型 API
（5 老师全部不可用时 fallback 到本地 Qwen2.5-72B），输出 .pt 蒸馏数据集，支持断点续传。

Usage:
  python -m scripts.data.build_distillset \
      --prompts data/prompts.json \
      --output outputs/distillset/distillset.pt \
      --teachers claude_opus_4_8,gpt_5_6_sol \
      --max-samples 100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sky_v1.training.teacher_client import (
    DistillSetBuilder,
    Qwen72BLocalTeacher,
    TeacherAPIClient,
)
from sky_v1.utils.logging import get_logger, setup_root_logger

log = get_logger("scripts.data.build_distillset")

# 内置默认蒸馏 prompt（未提供 --prompts 时使用），覆盖代码/推理/多模态/数学/写作等技能。
_DEFAULT_PROMPTS: list[str] = [
    "用 Python 实现一个快速排序，并解释其时间复杂度。",
    "请解释 Transformer 中多头注意力机制的工作原理。",
    "写一首关于秋天的现代诗，不少于八行。",
    "一只柴犬在太空漂浮的赛博朋克场景，请详细描述画面构图与配色。",
    "证明：对于任意正整数 n，n^3 - n 能被 6 整除。",
    "请把以下中文翻译成英文：科技的发展让人与人的距离变得更近。",
    "总结 RAG（检索增强生成）相比纯参数化模型的三个核心优势。",
    "设计一个三阶段训练方案（warmup/align/distill），简述每阶段目标。",
]


def _write_default_prompts() -> str:
    """将内置默认 prompt 写入临时 JSON 并返回路径。"""
    import json
    import tempfile
    fd, path = tempfile.mkstemp(prefix="sky_distill_prompts_", suffix=".json", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(_DEFAULT_PROMPTS, f, ensure_ascii=False, indent=2)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.data.build_distillset",
        description="构建 5 老师蒸馏数据集（+ Qwen2.5-72B 本地 fallback），输出 .pt 文件。",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="prompts 文件路径（JSON/CSV）；省略时使用内置默认 prompt 集",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/distillset/distillset.pt",
        help="输出 .pt 文件路径（默认 data/distillset/distillset.pt）",
    )
    parser.add_argument(
        "--teachers",
        type=str,
        default="all",
        help="逗号分隔的老师名（默认 all=全部 5 个老师）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制处理的 prompt 数量（默认全部）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="每个老师生成的最大 token 数（默认 512）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="teachers 配置 yaml 路径（默认 configs/training/teachers.yaml）",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        default=False,
        help="禁用本地 Qwen2.5-72B fallback",
    )
    return parser


def _resolve_teachers(arg: str, client: TeacherAPIClient) -> list[str] | None:
    if not arg or arg.strip().lower() == "all":
        return None
    names = [t.strip() for t in arg.split(",") if t.strip()]
    valid = set(client.teacher_names())
    bad = [n for n in names if n not in valid]
    if bad:
        log.warning("Unknown teacher names ignored", unknown=bad, valid=list(valid))
    return [n for n in names if n in valid] or None


def main() -> int:
    setup_root_logger("INFO")
    parser = _build_parser()
    try:
        args = parser.parse_args()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    config_path = args.config
    if config_path is None:
        default_cfg = ROOT / "configs" / "training" / "teachers.yaml"
        config_path = str(default_cfg) if default_cfg.exists() else None

    # 未提供 --prompts 时使用内置默认 prompt 集
    prompts_path = args.prompts
    if prompts_path is None:
        prompts_path = _write_default_prompts()
        print(f"[BuildDistillSet] 未指定 --prompts，使用内置默认 prompt 集（{len(_DEFAULT_PROMPTS)} 条）")

    client = TeacherAPIClient(config_path=config_path)
    teachers = _resolve_teachers(args.teachers, client)

    print("[BuildDistillSet] 配置:")
    print(f"  prompts      : {prompts_path}")
    print(f"  output       : {args.output}")
    print(f"  teachers     : {teachers or client.teacher_names()}")
    print(f"  max_samples  : {args.max_samples}")
    print(f"  max_tokens   : {args.max_tokens}")
    print(f"  fallback     : {not args.no_fallback}")
    print(f"  config_path  : {config_path}")

    builder = DistillSetBuilder(
        teacher_client=client,
        qwen_fallback=Qwen72BLocalTeacher(),
        config_path=config_path,
        use_fallback=not args.no_fallback,
    )

    try:
        stats = builder.build_from_prompts(
            prompts_file=prompts_path,
            output_path=args.output,
            max_samples=args.max_samples,
            teachers=teachers,
            max_tokens=args.max_tokens,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] build failed: {e}", file=sys.stderr)
        return 1

    print("[BuildDistillSet] 完成:")
    for k, v in stats.items():
        print(f"  {k:<16}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
