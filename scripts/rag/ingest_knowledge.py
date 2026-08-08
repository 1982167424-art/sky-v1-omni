"""CLI一键摄入sky_v1.rag.presets中的所有预置知识文档到Chroma/InMemory向量库。

Usage:
  python -m scripts.rag.ingest_knowledge [--config PATH] [--persist-dir PATH] [--backend inmemory|chroma] [--show-only] [--query STR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.rag.ingest_knowledge",
        description="一键摄入 sky_v1.rag.presets 中的所有预置知识文档到向量库。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/rag/vector_db_chroma.yaml",
        help="RAG 配置文件 YAML 路径 (默认: ./configs/rag/vector_db_chroma.yaml)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="覆盖配置中的向量库持久化目录 (默认: 使用 config 中的值)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["inmemory", "chroma"],
        default="chroma",
        help="向量库后端: chroma(持久化) 或 inmemory(内存, 默认chroma)",
    )
    parser.add_argument(
        "--show-only",
        action="store_true",
        default=False,
        help="仅打印 preset 列表，不执行实际摄入",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="若指定，摄入完成后立刻执行一次 query 并打印 top 5 结果",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not isinstance(args.config, str) or not args.config:
        raise argparse.ArgumentTypeError("--config 必须是非空字符串")
    if args.persist_dir is not None and (not isinstance(args.persist_dir, str) or not args.persist_dir):
        raise argparse.ArgumentTypeError("--persist-dir 必须是非空字符串或省略")
    if args.backend not in ("inmemory", "chroma"):
        raise argparse.ArgumentTypeError("--backend 必须是 'inmemory' 或 'chroma'")
    if not isinstance(args.show_only, bool):
        raise argparse.ArgumentTypeError("--show-only 必须是 flag (bool)")
    if args.query is not None and (not isinstance(args.query, str) or not args.query):
        raise argparse.ArgumentTypeError("--query 必须是非空字符串或省略")


def main() -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args()
        _validate_args(args)
    except (argparse.ArgumentError, argparse.ArgumentTypeError, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code == 0:
            return 0
        print(f"参数错误: {e}", file=sys.stderr)
        return 1

    try:
        from sky_v1.rag.presets import PRESET_DOCS
    except Exception as e:
        print(f"导入 sky_v1.rag.presets 失败: {e}", file=sys.stderr)
        return 1

    if args.show_only:
        print("[Preset 文档列表]")
        print(f"共 {len(PRESET_DOCS)} 个 preset:")
        for i, (category, doc_id, title, path) in enumerate(PRESET_DOCS, 1):
            print(f"  {i:2d}. [{category}] {doc_id} - {title}")
            print(f"       路径: {path}")
        return 0

    try:
        from sky_v1.rag import (
            KnowledgeBase,
            ChromaStore,
            InMemoryStore,
            SimEmbeddingFallback,
        )
    except Exception as e:
        print(f"导入 sky_v1.rag 核心模块失败: {e}", file=sys.stderr)
        return 1

    try:
        from sky_v1.utils.config import load_yaml_config
    except Exception:
        load_yaml_config = None

    persist_dir: str = "./chroma_data"
    if load_yaml_config is not None:
        try:
            cfg_path = Path(args.config)
            if cfg_path.exists():
                cfg = load_yaml_config(str(cfg_path))
                vdb = cfg.get("vector_db", {}) if isinstance(cfg, dict) else {}
                if isinstance(vdb, dict):
                    cfg_pd = vdb.get("persist_directory")
                    if isinstance(cfg_pd, str) and cfg_pd:
                        persist_dir = cfg_pd
        except Exception:
            pass

    if args.persist_dir is not None:
        persist_dir = args.persist_dir

    try:
        if args.backend == "inmemory":
            store = InMemoryStore()
        else:
            store = ChromaStore(persist_directory=persist_dir)
        embedder = SimEmbeddingFallback()
        kb = KnowledgeBase(store=store, embedder=embedder, persist_directory=persist_dir)
    except Exception as e:
        print(f"初始化 KnowledgeBase 失败: {e}", file=sys.stderr)
        return 1

    try:
        stats = kb.ingest_presets()
    except Exception as e:
        print(f"ingest_presets 失败: {e}", file=sys.stderr)
        return 1

    total_docs = stats.get("total_docs", 0)
    total_chunks = stats.get("total_chunks", 0)
    categories = stats.get("categories", {})
    print("[摄入完成]")
    print(f"  total_docs   : {total_docs}")
    print(f"  total_chunks : {total_chunks}")
    print(f"  categories   : {dict(categories)}")

    if args.query is not None:
        print()
        print(f"[查询: {args.query!r}]")
        try:
            results = kb.query(args.query, 5)
        except Exception as e:
            print(f"查询失败: {e}", file=sys.stderr)
            return 1
        if not results:
            print("  (无结果)")
        else:
            for i, r in enumerate(results, 1):
                rid = r.get("id", "")
                dist = r.get("distance", 0.0)
                meta = r.get("metadata") or {}
                title = meta.get("title", "")
                doc = r.get("document", "") or ""
                preview = doc[:150].replace("\n", " ")
                print(f"  结果 #{i}:")
                print(f"    id       : {rid}")
                print(f"    distance : {dist:.6f}")
                print(f"    title    : {title}")
                print(f"    document : {preview}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
