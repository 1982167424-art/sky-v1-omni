"""CLI一键启动 sky-v1 Agent API服务（uvicorn.run）。

Usage:
  python -m scripts.agent.start_agent_server [--config-dir DIR] [--host HOST] [--port PORT] [--log-level LEVEL] [--log-file FILE] [--ingest-presets] [--no-reload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.agent.start_agent_server",
        description="一键启动 sky-v1 Agent API 服务 (FastAPI + uvicorn)。",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./configs",
        help="配置文件目录 (默认: ./configs)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务绑定 host (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务绑定 port (默认: 8000, 范围 1-65535)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="日志级别: DEBUG/INFO/WARNING/ERROR/CRITICAL (默认: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="可选：日志文件路径 (默认: 仅控制台)",
    )
    parser.add_argument(
        "--ingest-presets",
        action="store_true",
        default=False,
        help="启动时立即对 rag_kb 调用 ingest_presets()",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        default=False,
        help="关闭 uvicorn --reload (开发时默认打开 reload)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not isinstance(args.config_dir, str) or not args.config_dir:
        raise argparse.ArgumentTypeError("--config-dir 必须是非空字符串")
    if not isinstance(args.host, str) or not args.host:
        raise argparse.ArgumentTypeError("--host 必须是非空字符串")
    if not isinstance(args.port, int) or isinstance(args.port, bool):
        raise argparse.ArgumentTypeError("--port 必须是整数")
    if args.port < 1 or args.port > 65535:
        raise ValueError("--port 必须在 1 到 65535 之间")
    if not isinstance(args.log_level, str) or not args.log_level:
        raise argparse.ArgumentTypeError("--log-level 必须是非空字符串")
    if args.log_file is not None and (not isinstance(args.log_file, str) or not args.log_file):
        raise argparse.ArgumentTypeError("--log-file 必须是非空字符串或省略")
    if not isinstance(args.ingest_presets, bool):
        raise argparse.ArgumentTypeError("--ingest-presets 必须是 flag (bool)")
    if not isinstance(args.no_reload, bool):
        raise argparse.ArgumentTypeError("--no-reload 必须是 flag (bool)")


def _print_banner(version: str, host: str, port: int) -> None:
    banner = r"""
   _____ _          ____  _         ___  _                       _ 
  / ____| |        / __ \| |       / _ \| |                     (_)
 | (___ | | ___   _| |  | | |     | | | | |_ ___  _ __ ___  _ __  _ 
  \___ \| |/ / | | | |  | | |     | | | | __/ _ \| '_ ` _ \| '_ \| |
  ____) |   <| |_| | |__| | |____ | |_| | || (_) | | | | | | | | | |
 |_____/|_|\_\\__, |\____/|______| \___/ \__\___/|_| |_| |_|_| |_|_|
               __/ |                                                
              |___/                                                 
"""
    print(banner)
    print(f"  sky-v1-omni agent M1  |  version {version}")
    print("-" * 72)
    print(f"  Health  : http://{host}:{port}/health")
    print(f"  Docs    : http://{host}:{port}/docs")
    print(f"  OpenAPI : http://{host}:{port}/openapi.json")
    print("-" * 72)


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
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        return 1

    log_level_raw = str(args.log_level).upper()
    log_level = log_level_raw if log_level_raw in VALID_LOG_LEVELS else "INFO"
    if log_level != log_level_raw:
        print(
            f"WARNING: 无效 log-level {log_level_raw!r}, 回退到 INFO",
            file=sys.stderr,
        )

    try:
        from sky_v1.utils.logging import setup_root_logger
    except ImportError as e:
        print(f"缺少依赖: 无法导入 sky_v1.utils.logging.setup_root_logger: {e}", file=sys.stderr)
        print("请确认已 pip install sky-v1-omni 的依赖 (fastapi, uvicorn 等)", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"初始化日志模块失败: {e}", file=sys.stderr)
        return 1

    try:
        setup_root_logger(level=log_level, log_file=args.log_file)
    except Exception as e:
        print(f"setup_root_logger 失败 (非致命): {e}", file=sys.stderr)

    try:
        from sky_v1 import __version__
        from sky_v1.rag import KnowledgeBase
        from sky_v1.agent import SkyAgent
        from sky_v1.api.app import create_app
    except ImportError as e:
        print(f"缺少依赖: 导入核心模块失败: {e}", file=sys.stderr)
        print("请确认已 pip install sky-v1-omni 的依赖 (fastapi, uvicorn, chromadb 等)", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"导入核心模块失败: {e}", file=sys.stderr)
        return 1

    try:
        import uvicorn
    except ImportError as e:
        print(f"缺少依赖: 无法导入 uvicorn: {e}", file=sys.stderr)
        print("请执行: pip install uvicorn", file=sys.stderr)
        return 1

    config_dir_path = Path(args.config_dir).resolve()
    chroma_data_path = (config_dir_path.parent / "chroma_data").resolve()
    persist_directory = str(chroma_data_path)

    rag_kb: Any = None
    try:
        rag_kb = KnowledgeBase(persist_directory=persist_directory)
    except Exception as e:
        print(f"WARNING: KnowledgeBase 初始化失败，将使用 None: {e}", file=sys.stderr)
        rag_kb = None

    if args.ingest_presets and rag_kb is not None:
        try:
            stats = rag_kb.ingest_presets()
            total_docs = stats.get("total_docs", 0)
            total_chunks = stats.get("total_chunks", 0)
            print(f"[ingest-presets] 完成: docs={total_docs}, chunks={total_chunks}")
        except Exception as e:
            print(f"WARNING: ingest_presets 失败: {e}", file=sys.stderr)

    try:
        agent = SkyAgent(rag_kb=rag_kb)
    except Exception as e:
        print(f"WARNING: SkyAgent 初始化失败，将使用 None: {e}", file=sys.stderr)
        agent = None

    try:
        app = create_app(
            config_dir=str(config_dir_path),
            rag_kb=rag_kb,
            agent=agent,
        )
    except Exception as e:
        print(f"create_app 失败: {e}", file=sys.stderr)
        return 1

    _print_banner(__version__, args.host, args.port)

    reload_flag = not bool(args.no_reload)
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=reload_flag,
            log_level=log_level.lower(),
            access_log=True,
        )
    except Exception as e:
        print(f"uvicorn.run 启动失败: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
