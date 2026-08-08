"""同步 13 个 GitHub 仓库到 RAG 知识库（无 GitHub API Key 时使用静态 preset 数据，绝不崩）。

Usage:
  python -m scripts.data.sync_github_repos_to_rag [--kb-persist-dir PATH] [--github-token-env GITHUB_TOKEN] [--owner 1982167424-art] [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Any


FALLBACK_STATIC_REPOS: list[dict[str, Any]] = [
    {
        "name": "sky-v1-omni",
        "description": "5-modal Omni Model 主仓库（文本/图像/3D/视频/音频）",
        "language": "Python",
        "stars": 42,
        "readme": (
            "# sky-v1-omni\n\n"
            "sky-v1 0.1.0-M1 里程碑核心仓库。\n\n"
            "实现 5-modal（Text/Image/3D/Video/Audio）统一大模型的训练脚手架、\n"
            "RAG 知识库子系统（sky_v1/rag/）、OpenAI 兼容 API Server（sky_v1/api/）、\n"
            "Agent Loop（sky_v1/agent/）、CLI 工具（scripts/）。\n\n"
            "关键模块: sky_v1.utils（日志、配置、种子、重试）、sky_v1.rag（RAG 子系统）、\n"
            "sky_v1.models（统一骨干）、sky_v1.training（三阶段训练）、\n"
            "sky_v1.api（FastAPI 服务）、sky_v1.agent（Agent + MCP Skill）。\n"
        ),
    },
    {
        "name": "IKNOW",
        "description": "数据集构建、知识增强与评测工具包",
        "language": "Python",
        "stars": 28,
        "readme": (
            "# IKNOW\n\n"
            "IKNOW = Intelligent Knowledge for NLP mOdels Working。\n\n"
            "数据侧平台仓库，负责从网页、GitHub、论文、电子书、多模态数据源\n"
            "爬取、清洗、去重、合成指令对、构造偏好数据集。\n\n"
            "主要管线: Common Crawl → 语言分类 → 去重（MinHash LSH）→ 质量打分器\n"
            "→ 困惑度过滤 → 最终预训练语料；Self-Instruct + Evol-Instruct 合成 SFT 数据；\n"
            "基于 ORCA / UltraFeedback 的 5 老师偏好蒸馏 → DPO 数据。\n"
        ),
    },
    {
        "name": "pppdlw",
        "description": "大规模并行训练的轻量封装（pp + deepspeed + lightning w）",
        "language": "Python",
        "stars": 19,
        "readme": (
            "# pppdlw\n\n"
            "pppdlw = PyTorch + PP (Pipeline Parallel) + DeepSpeed + Lightning Wrapper。\n\n"
            "大规模分布式训练框架的轻量封装。\n\n"
            "核心能力: ZeRO-1/2/3 配置化；张量并行（Megatron-LM）集成；\n"
            "流水线并行 GPipe 调度；混合精度 BF16/FP8；FlashAttention v2/v3 自动切换；\n"
            "Checkpoint 断点续训；wandb / tensorboard 双写。\n"
        ),
    },
    {
        "name": "ppdbeta",
        "description": "pppdlw 的 Web 控制台，可视化 loss、激活、样本",
        "language": "JavaScript",
        "stars": 12,
        "readme": (
            "# ppdbeta\n\n"
            "pppdlw 的 Web Dashboard（Beta 版）。Node.js + Vue 3 实现，\n"
            "WebSocket 实时推送训练指标。\n\n"
            "功能: Loss 曲线、学习率曲线、梯度范数、激活分布直方图、\n"
            "每 N 步随机采样生成样例、Bad Case 可视化、节点资源热力图、\n"
            "超参对比面板（支持最多 8 次跑批 overlay 对比）。\n"
        ),
    },
    {
        "name": "SKYOS",
        "description": "skyOS 下一代操作系统 Web Shell（SKYOS V1）",
        "language": "JavaScript",
        "stars": 35,
        "readme": (
            "# SKYOS V1\n\n"
            "SKYOS = sky-v1 Operating System。\n\n"
            "基于浏览器的下一代 AI 原生操作系统 Web Shell，\n"
            "把应用变成智能体可组合的 Skill 包。\n\n"
            "核心特性: 类 Nushell 的命令行，每个命令是一个 MCP Tool，\n"
            "支持管道组合；多窗口 Web Desktop，跨 App 拖拽由 Agent Kernel 做语义级解释；\n"
            "Agent Kernel 通过 WebSocket 连接 omni 的 Agent API Server。\n"
        ),
    },
    {
        "name": "SKY-OS",
        "description": "skyOS V0 原型，Shell scripts + 桌面 UI 概念验证",
        "language": "Shell",
        "stars": 8,
        "readme": (
            "# SKY-OS (V0 原型)\n\n"
            "SKYOS V0 原型仓库。用 Bash + 简单的 Electron Shell + Linux Namespace 技术\n"
            "做的早期 PoC，验证 AI 原生命令行 + 容器化隔离概念。\n\n"
            "状态: 进入只读归档阶段，不再接受新功能 PR；\n"
            "仅在需要参考历史设计决策时查阅。\n"
        ),
    },
    {
        "name": "1982167424-art",
        "description": "sky-v1 团队核心成员 1982167424 的个人主页、作品集、模型演示",
        "language": "Markdown",
        "stars": 15,
        "readme": (
            "# 1982167424-art\n\n"
            "sky-v1 核心贡献者 @1982167424 的个人作品集仓库。\n\n"
            "纯静态 GitHub Pages + README Gallery。\n\n"
            "内容: 该成员产出的图像生成样例（Stable Diffusion / FLUX 调参实验）、\n"
            "3D 作品（Blender + 3DGS 重建）、视频生成样例（Sora/CogVideo）、\n"
            "模型蒸馏对比表格、技术博客译文。\n"
        ),
    },
    {
        "name": "STAR_CHAT",
        "description": "AI 聊天助手前端（基于小米 mimo SDK + 多模型路由）",
        "language": "TypeScript",
        "stars": 22,
        "readme": (
            "# STAR_CHAT\n\n"
            "StarChat = sky-v1 Team's AI Chat。\n\n"
            "基于小米 mimo 前端 SDK（React + TypeScript）+ 自研多模型路由后端的 AI 聊天产品。\n\n"
            "特性: 多模型切换（omni 主模型、Qwen、Llama 3 作为兜底）、\n"
            "System Prompt 预设商城（100+ 角色预设）、\n"
            "多模态上传（图片/音频/文档）→ RAG 检索、\n"
            "对话导出 Markdown、协同编辑（多人共享同一会话链接）。\n"
        ),
    },
    {
        "name": "STAR-CHAT",
        "description": "STAR_CHAT 主仓库的镜像/二次开发分支（历史兼容仓库）",
        "language": "TypeScript",
        "stars": 5,
        "readme": (
            "# STAR-CHAT (镜像)\n\n"
            "历史镜像仓库。\n\n"
            "由于组织迁移（从个人账户 org 搬到 sky-v1 org）时 fork 关系丢失\n"
            "而保留的兼容只读镜像。\n\n"
            "内容: 复刻 V0 版本的代码提交历史，issue/PR 关闭状态与主仓一致。\n"
        ),
    },
    {
        "name": "tanshi-app",
        "description": "叹思 App：轻量移动端 AI 应用，1 星起步阶段",
        "language": "TypeScript",
        "stars": 3,
        "readme": (
            "# tanshi-app\n\n"
            "叹思（TanSi）轻量移动端 AI 应用。\n\n"
            "Capacitor + Ionic + TypeScript，一套代码打包 iOS / Android / 小程序。\n\n"
            "当前状态: 立项早期（1 星阶段），已打通 omni API 流式对话 + \n"
            "Whisper cpp 端上 ASR + 端上 HiFi-GAN Lite TTS 最小闭环。\n"
        ),
    },
    {
        "name": "starbuddy-agent",
        "description": "StarBuddy Agent Tooling：MCP Skills 与 Agent 工作流编排",
        "language": "Python",
        "stars": 18,
        "readme": (
            "# starbuddy-agent\n\n"
            "StarBuddy = sky-v1 的 Agent 工具 SDK。\n\n"
            "严格实现 MCP（Model Context Protocol）Server/Client 标准，\n"
            "内置 50+ 通用 Agent Skill。\n\n"
            "内置工具类: 通用（文件读写、Shell 执行、Web 抓取、代码沙箱）、\n"
            "数据（Google Sheets / Lark Base / Notion Database 读写）、\n"
            "日程（Google Calendar / Lark Calendar / Outlook 日程创建与查询）、\n"
            "消息（Lark / 企业微信 / Slack / Telegram 消息发送）、\n"
            "RAG：sky_v1.rag 子系统的 MCP 包装（增删查知识库）。\n"
        ),
    },
    {
        "name": "starbuddy-code",
        "description": "StarBuddy Code Tooling：代码库索引、AST 检索、Diff 生成",
        "language": "Python",
        "stars": 14,
        "readme": (
            "# starbuddy-code\n\n"
            "StarBuddy Code = 面向软件开发 Agent 的代码理解工具集。\n\n"
            "核心能力: 基于 Tree-sitter 的多语言（Python/TS/Go/Rust/Java/C++）AST 索引，\n"
            "支持查找符号定义/引用/调用图；仓库级 Semantic Search：用 bge-m3 对每个函数、类、方法做离线嵌入；\n"
            "PR 级 Diff Review Agent：结合 Brooks Lint（12 本经典工程书规则）输出结构化 Review 评论；\n"
            "Commit Message 生成、Git Worktree 多分支并行 Agent 执行。\n"
        ),
    },
    {
        "name": "ai-introduction",
        "description": "面向中学生的 AI 科普网站，含 1 个 issue",
        "language": "HTML",
        "stars": 6,
        "readme": (
            "# ai-introduction\n\n"
            "面向中学生的 AI 科普网站。\n\n"
            "纯静态 HTML + CSS + 一点点 JS，无构建工具。\n\n"
            "内容: 什么是 AI、什么是机器学习、什么是神经网络、\n"
            "大语言模型原理（漫画图解）、Stable Diffusion 动手小实验、\n"
            "AI 伦理、sky-v1 团队介绍。\n"
        ),
    },
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.data.sync_github_repos_to_rag",
        description="同步指定 owner 下的 GitHub 仓库到 RAG 知识库（无 token 时自动 fallback 到静态数据）。",
    )
    parser.add_argument(
        "--kb-persist-dir",
        type=str,
        default=None,
        help="知识库持久化目录 (默认: 项目根下 ./chroma_data)",
    )
    parser.add_argument(
        "--github-token-env",
        type=str,
        default="GITHUB_TOKEN",
        help="存放 GitHub Token 的环境变量名 (默认: GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--owner",
        type=str,
        default="1982167424-art",
        help="GitHub 仓库 owner 用户名或组织名 (默认: 1982167424-art)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="不入库，仅打印将摄入的仓库列表与内容",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制同步的仓库数量 (默认: 全部)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.kb_persist_dir is not None and (not isinstance(args.kb_persist_dir, str) or not args.kb_persist_dir):
        raise argparse.ArgumentTypeError("--kb-persist-dir 必须是非空字符串或省略")
    if not isinstance(args.github_token_env, str) or not args.github_token_env:
        raise argparse.ArgumentTypeError("--github-token-env 必须是非空字符串")
    if not isinstance(args.owner, str) or not args.owner:
        raise argparse.ArgumentTypeError("--owner 必须是非空字符串")
    if not isinstance(args.dry_run, bool):
        raise argparse.ArgumentTypeError("--dry-run 必须是 flag (bool)")
    if args.limit is not None:
        if not isinstance(args.limit, int) or isinstance(args.limit, bool) or args.limit < 1:
            raise argparse.ArgumentTypeError("--limit 必须是 >=1 的整数或省略")


def _try_fetch_github_repos(owner: str, token_env_name: str) -> tuple[list[dict[str, Any]], bool]:
    """尝试通过 GitHub API 拉真实仓库列表。返回 (repos, ok)。ok=False 表示需 fallback。"""
    fallback_used = True
    repos: list[dict[str, Any]] = []

    token = os.environ.get(token_env_name, "")
    if not isinstance(token, str) or not token.strip():
        return [], False

    try:
        import httpx
    except ImportError:
        return [], False

    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        list_url = f"https://api.github.com/users/{owner}/repos?per_page=100"
        resp = httpx.get(list_url, headers=headers, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return [], False
        raw_repos = [r for r in data if isinstance(r, dict)]
    except (httpx.HTTPError, ValueError, Exception):
        return [], False

    try:
        for rr in raw_repos:
            try:
                name = rr.get("name")
                if not isinstance(name, str) or not name:
                    continue
                description = rr.get("description") or ""
                if not isinstance(description, str):
                    description = ""
                language = rr.get("language") or "Unknown"
                if not isinstance(language, str):
                    language = "Unknown"
                stars = rr.get("stargazers_count") or 0
                if not isinstance(stars, int):
                    stars = 0

                readme_text = ""
                try:
                    readme_url = f"https://api.github.com/repos/{owner}/{name}/readme"
                    rresp = httpx.get(readme_url, headers=headers, timeout=15.0)
                    if rresp.status_code == 200:
                        rdata = rresp.json()
                        if isinstance(rdata, dict):
                            b64 = rdata.get("content") or ""
                            if isinstance(b64, str) and b64:
                                try:
                                    readme_text = base64.b64decode(b64).decode("utf-8", errors="ignore")
                                except Exception:
                                    readme_text = ""
                except Exception:
                    readme_text = ""

                repos.append({
                    "name": name,
                    "description": description,
                    "language": language,
                    "stars": stars,
                    "readme": readme_text,
                })
            except Exception:
                continue
        fallback_used = False
    except Exception:
        return [], False

    return repos, not fallback_used


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

    owner = args.owner

    repos: list[dict[str, Any]] = []
    fallback_used = False
    try:
        repos, ok = _try_fetch_github_repos(owner, args.github_token_env)
        if not ok or not repos:
            fallback_used = True
            repos = list(FALLBACK_STATIC_REPOS)
            print("[WARNING] 使用 FALLBACK_STATIC_REPOS 静态列表（GitHub API 不可用或无 token）")
    except Exception as e:
        fallback_used = True
        repos = list(FALLBACK_STATIC_REPOS)
        print(f"[WARNING] GitHub 拉取异常，使用静态 fallback: {e}")

    if args.limit is not None and isinstance(args.limit, int) and args.limit > 0:
        repos = repos[: args.limit]

    if args.dry_run:
        print(f"[DRY-RUN] 将同步 {len(repos)} 个仓库 (owner={owner!r}, fallback_used={fallback_used})")
        for i, r in enumerate(repos, 1):
            name = r.get("name", "")
            desc = r.get("description", "")
            lang = r.get("language", "")
            stars = r.get("stars", 0)
            readme = r.get("readme", "") or ""
            print(f"  #{i:2d}. {name}")
            print(f"       描述: {desc}")
            print(f"       语言: {lang}  Stars: {stars}")
            print(f"       README预览: {readme[:120].replace(chr(10), ' ')}...")
        return 0

    persist_dir = args.kb_persist_dir
    if not persist_dir:
        project_root = Path(__file__).resolve().parents[2]
        persist_dir = str((project_root / "chroma_data").resolve())

    try:
        from sky_v1.rag import KnowledgeBase, Document
    except ImportError as e:
        print(f"缺少依赖: 无法导入 sky_v1.rag: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"导入 sky_v1.rag 失败: {e}", file=sys.stderr)
        return 1

    try:
        kb = KnowledgeBase(persist_directory=persist_dir)
    except Exception as e:
        print(f"初始化 KnowledgeBase 失败: {e}", file=sys.stderr)
        return 1

    docs: list[Any] = []
    for r in repos:
        try:
            name = r.get("name") or ""
            if not isinstance(name, str) or not name:
                continue
            description = r.get("description") or ""
            if not isinstance(description, str):
                description = ""
            language = r.get("language") or "Unknown"
            if not isinstance(language, str):
                language = "Unknown"
            stars = r.get("stars") or 0
            if not isinstance(stars, int):
                stars = 0
            readme = r.get("readme") or ""
            if not isinstance(readme, str):
                readme = ""

            text = (
                f"# {name}\n"
                f"描述: {description}\n"
                f"语言: {language}\n"
                f"Stars: {stars}\n\n"
                f"{readme}"
            )
            metadata = {
                "category": "github_repos",
                "repo": name,
                "owner": owner,
                "source": "github_sync",
            }
            doc = Document(
                id=f"github_{name}",
                text=text,
                metadata=metadata,
                category="github_repos",
            )
            docs.append(doc)
        except Exception as e:
            print(f"[WARNING] 构造 repo 文档失败，跳过: {e}", file=sys.stderr)
            continue

    total_chunks = 0
    success = 0
    try:
        if docs:
            stats = kb.ingest_documents(docs)
            total_chunks = stats.get("total_chunks", 0) if isinstance(stats, dict) else 0
            success = stats.get("total_docs", 0) if isinstance(stats, dict) else 0
        else:
            success = 0
            total_chunks = 0
    except Exception as e:
        print(f"[WARNING] kb.ingest_documents 异常: {e}", file=sys.stderr)
        success = 0
        total_chunks = 0

    print("[同步完成]")
    print(f"  success        : {success}")
    print(f"  fallback_used  : {fallback_used}")
    print(f"  total_chunks   : {total_chunks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
