"""sky CLI: chat / embed / serve / train / rag.

Run:
  sky --help
  sky chat "What is sky-v1?"
  sky serve --host 0.0.0.0 --port 8000
  sky train --phase toy --steps 10
  sky rag ingest docs/
  sky rag query "How does UniTransformer work?"
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="sky-v1-omni: 5-modal AI model + Agent CLI",
    rich_markup_mode="markdown",
)


@app.command("chat")
def cmd_chat(
    message: str = typer.Argument(..., help="User message"),
    model: str = typer.Option("mini-sdk", help="Model name / config YAML"),
    mode: str = typer.Option("direct", help="direct or http"),
    base_url: str = typer.Option("http://localhost:8000/v1", help="HTTP base URL"),
    max_new_tokens: int = typer.Option(64, help="Max new tokens"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
) -> None:
    """Start (or send single message to) sky-v1 chat."""
    from ..sdk.client import SkySDK
    sdk = SkySDK(engine=mode, base_url=base_url, model_name=model)
    resp = sdk.chat_completions(
        messages=[{"role": "user", "content": message}],
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    typer.echo(resp["choices"][0]["message"]["content"])


@app.command("embed")
def cmd_embed(
    texts: list[str] = typer.Argument(..., help="Text(s) to embed"),
    model: str = typer.Option("mini-sdk"),
    output: Optional[Path] = typer.Option(None, "--out", "-o", help="Write JSON to file"),
) -> None:
    """Get embeddings for one or more texts."""
    from ..sdk.client import SkySDK
    import json as _json
    sdk = SkySDK(engine="direct", model_name=model)
    out = sdk.embeddings(list(texts))
    txt = _json.dumps(out, indent=2, default=str)
    if output:
        output.write_text(txt)
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(txt)


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Launch FastAPI inference/agent server (Uvicorn)."""
    import uvicorn
    uvicorn.run(
        "sky_v1.api.app:create_app",
        host=host, port=port, reload=reload, factory=True,
    )


@app.command("train")
def cmd_train(
    phase: str = typer.Option("toy", help="phase1 / phase2 / phase3 / toy"),
    steps: int = typer.Option(10, help="Steps for toy phase"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run training (toy smoke-test or phase scripts)."""
    if phase == "toy":
        from subprocess import Popen
        cmd = [sys.executable, "-m", "scripts.training.train_toy_overfit", "--steps", str(steps), "--device", "cpu"]
        with Popen(cmd, cwd=str(Path(__file__).resolve().parents[2])) as p:
            sys.exit(p.wait() or 0)
    else:
        script_map = {
            "phase1": "scripts.training.phase1_warmup",
            "phase2": "scripts.training.phase2_align",
            "phase3": "scripts.training.phase3_distill",
        }
        if phase not in script_map:
            typer.echo(f"Unknown phase: {phase}", err=True)
            raise typer.Exit(2)
        cmd = [sys.executable, "-m", script_map[phase]]
        if config:
            cmd += ["--config", str(config)]
        from subprocess import Popen
        with Popen(cmd, cwd=str(Path(__file__).resolve().parents[2])) as p:
            sys.exit(p.wait() or 0)


@app.command("rag")
def cmd_rag(
    action: str = typer.Argument(..., help="ingest | query | list"),
    arg: Optional[str] = typer.Argument(None, help="query string or path"),
) -> None:
    """RAG knowledge-base operations."""
    from ..rag.knowledge_base import SkyKnowledgeBase
    kb = SkyKnowledgeBase()
    if action == "ingest" and arg:
        p = Path(arg)
        if p.is_file():
            kb.ingest_file(p)
            typer.echo(f"Ingested {p}")
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    kb.ingest_file(f)
            typer.echo("Ingested dir")
        else:
            typer.echo(f"Not found: {p}", err=True)
            raise typer.Exit(2)
    elif action == "query" and arg:
        hits = kb.search(arg, k=5)
        for h in hits:
            score = h.get("score")
            s = f"{score:.3f}" if isinstance(score, float) else str(score)
            typer.echo(f"- [{s}] {str(h.get('title',''))[:80]}")
    elif action == "list":
        typer.echo(f"docs_in_index: {kb.count()}")
    else:
        typer.echo("Usage: sky rag ingest <path> | query <q> | list", err=True)
        raise typer.Exit(2)


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="搜索关键词"),
    num: int = typer.Option(5, "--num", "-n", min=1, max=20, help="返回条数"),
    no_cache: bool = typer.Option(False, "--no-cache", help="跳过缓存"),
    providers: str = typer.Option(
        "google,baidu,bing,toutiao",
        "--providers",
        "-p",
        help="引擎白名单，逗号分隔：google/baidu/bing/toutiao/tavily",
    ),
    allow_simulated: bool = typer.Option(
        False,
        "--allow-simulated",
        help="[开发模式] 真实引擎全失败时允许返回模拟结果",
    ),
) -> None:
    """毫秒级联网搜索（Google/百度/Bing/今日头条 并发 + TTL/LRU 缓存）。"""
    from ..agent.base import ToolContext
    from ..agent.tools.search_tools import WebSearchTool

    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    tool = WebSearchTool()
    ctx = ToolContext(session_id="cli-search")
    res = tool.run(
        ctx,
        query=query,
        num_results=num,
        providers=provider_list if provider_list else None,
        skip_cache=no_cache,
        allow_simulated=allow_simulated,
    )
    typer.echo(res.output)


@app.command("think")
def cmd_think(
    question: str = typer.Argument(..., help="需要深度推理的问题"),
    steps: int = typer.Option(3, "--steps", "-s", min=1, max=6, help="最大推理步数"),
    no_web: bool = typer.Option(False, "--no-web", help="不调用联网搜索"),
    no_cite: bool = typer.Option(False, "--no-cite", help="省略引用标注"),
) -> None:
    """深度多步推理（Plan-Act-Observe-Reflect + 置信度）。"""
    from ..agent.base import ToolContext
    from ..rag.knowledge_base import SkyKnowledgeBase
    from ..agent.tools.search_tools import DeepReasoningTool
    tool = DeepReasoningTool()
    try:
        kb = SkyKnowledgeBase()
    except Exception:
        kb = None
    ctx = ToolContext(session_id="cli-think", rag_kb=kb)
    res = tool.run(
        ctx,
        question=question,
        max_iterations=steps,
        enable_web_search=not no_web,
        citations_needed=not no_cite,
    )
    typer.echo(res.output)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
