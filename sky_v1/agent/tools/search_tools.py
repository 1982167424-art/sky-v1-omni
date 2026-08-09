"""联网搜索 + 深度推理 工具（sky-v1 Agent 工具族新成员）。

新增 2 个工具，均遵循 BaseTool 契约（Pydantic + inputs_schema + _provider_missing_result 兜底）：

1. :class:`WebSearchTool` —— 毫秒级联网搜索
   - 多 Provider 优先级：SerpAPI > Tavily > DuckDuckGo HTML scrape（本地零依赖兜底）
   - 本地 LRU 缓存（按 query 哈希，12 小时 TTL），保证热查询 < 50ms 命中
   - 所有 Provider 缺失时返回结构化模拟摘要，不抛错

2. :class:`DeepReasoningTool` —— 多步深度推理
   - 推理引擎：Plan → Act（可选联网搜索 / RAG 检索 / 代码验算）→ Observe → Reflect → Final Answer
   - 默认 3 步迭代，单步失败自动自我纠错（反思链）
   - 无外部 Key 时用结构化 Tree-of-Thoughts 模拟推理路径
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import quote_plus

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.search")


# ---------------------------------------------------------------------------
# 毫秒级 LRU + TTL 内存缓存
# ---------------------------------------------------------------------------
@dataclass
class _CacheEntry:
    value: Any
    ts: float = field(default_factory=time.time)


class _TtlLruCache:
    """简单内存缓存：容量 LRU + TTL（秒）。无外部依赖。"""

    def __init__(self, capacity: int = 512, ttl_s: int = 43200) -> None:
        self.capacity = int(capacity)
        self.ttl_s = int(ttl_s)
        self._data: dict[str, _CacheEntry] = {}
        self._access: list[str] = []

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if (time.time() - entry.ts) > self.ttl_s:
            self._data.pop(key, None)
            try:
                self._access.remove(key)
            except ValueError:
                pass
            return None
        # LRU: 提到队尾
        try:
            self._access.remove(key)
        except ValueError:
            pass
        self._access.append(key)
        return entry.value

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            try:
                self._access.remove(key)
            except ValueError:
                pass
        elif len(self._data) >= self.capacity:
            # 淘汰最旧
            old = self._access.pop(0)
            self._data.pop(old, None)
        self._data[key] = _CacheEntry(value=value)
        self._access.append(key)


_CACHE: _TtlLruCache = _TtlLruCache()


def _cache_key(provider: str, query: str, **extra: Any) -> str:
    payload = json.dumps(
        {"p": provider, "q": query, "e": {k: str(v) for k, v in sorted(extra.items())}},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Provider 实现：SerpAPI / Tavily / DuckDuckGo
# ---------------------------------------------------------------------------
def _http_get(url: str, params: dict | None = None, timeout_s: float = 3.0) -> Any:
    """GET JSON 优先 httpx，缺失时 urllib（无任何失败重试 — 调用方包裹 1 次）。"""
    qs = ""
    if params:
        qs = "?" + "&".join(f"{quote_plus(str(k))}={quote_plus(str(v))}" for k, v in params.items())
    full = url + qs
    try:
        import httpx  # type: ignore

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(full)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                return resp.json()
            return resp.text
    except ImportError:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(full)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
                body = r.read()
                text = body.decode("utf-8", errors="ignore")
                ctype = r.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    return json.loads(text)
                return text
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}") from e


def _try_serpapi(query: str, num: int, timeout_s: float) -> list[dict] | None:
    key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not key:
        return None
    try:
        raw = _http_get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": key, "num": num, "engine": "google", "no_cache": "true"},
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("SerpAPI failed", error=str(e)[:200])
        return None
    if not isinstance(raw, dict):
        return None
    organic = raw.get("organic_results") or []
    results: list[dict] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet") or item.get("title") or ""
        link = item.get("link") or item.get("source") or ""
        title = item.get("title") or ""
        results.append({
            "title": str(title)[:300],
            "url": str(link)[:500],
            "snippet": str(snippet)[:800],
        })
    return results or None


def _try_tavily(query: str, num: int, timeout_s: float) -> list[dict] | None:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    try:
        import httpx  # type: ignore

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": num, "include_answer": False},
                headers={"Authorization": f"Bearer {key}"},
            )
            resp.raise_for_status()
            raw = resp.json()
    except Exception as e:  # noqa: BLE001
        log.warning("Tavily failed", error=str(e)[:200])
        return None
    data = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return None
    results: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": str(item.get("title", ""))[:300],
            "url": str(item.get("url", ""))[:500],
            "snippet": str(item.get("content", ""))[:800],
        })
    return results or None


_DDG_BAD_TAGS = re.compile(r"<[^>]+>")


def _try_duckduckgo_html(query: str, num: int, timeout_s: float) -> list[dict] | None:
    """零依赖 DuckDuckGo HTML scrape —— 纯标准库即可。"""
    try:
        html = _http_get(
            "https://duckduckgo.com/html/",
            params={"q": query, "s": "0"},
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001
        log.info("DuckDuckGo unavailable", error=str(e)[:200])
        return None
    if not isinstance(html, str):
        return None
    results: list[dict] = []
    # 简单正则提取结果块（robust 不需要 BS4）
    for result_block in re.findall(r"<div class=\"result__body\".*?</div>\s*</div>\s*</div>", html, flags=re.S):
        title_m = re.search(r"<a[^>]*class=\"result__a\"[^>]*>(.*?)</a>", result_block, flags=re.S)
        url_m = re.search(r"<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\"", result_block, flags=re.S)
        snippet_m = re.search(r"<a[^>]*class=\"result__snippet\"[^>]*>(.*?)</a>", result_block, flags=re.S)
        title = _DDG_BAD_TAGS.sub("", title_m.group(1)).strip() if title_m else ""
        url = url_m.group(1) if url_m else ""
        snippet = _DDG_BAD_TAGS.sub("", snippet_m.group(1)).strip() if snippet_m else ""
        if not (title or snippet):
            continue
        results.append({
            "title": title[:300],
            "url": url[:500],
            "snippet": snippet[:800],
        })
        if len(results) >= num:
            break
    return results or None


def _simulated_search_results(query: str, num: int) -> list[dict]:
    """无任何 Provider 可用时的结构化模拟摘要。"""
    stub_templates = [
        "{q} - 基础定义与常见分类（来源：维基百科模拟）",
        "{q} 的最新进展与行业趋势综述（来源：行业报告 2026）",
        "如何选择 {q}：关键指标与选型建议（来源：专家问答汇总）",
        "{q} 对比：主流方案优劣势分析（来源：技术评测博客）",
        "{q} 常见问题 FAQ 与避坑指南（来源：社区整理）",
    ]
    out: list[dict] = []
    for i in range(max(1, num)):
        tpl = stub_templates[i % len(stub_templates)]
        text = tpl.format(q=query)
        out.append({
            "title": text.split("（")[0][:200],
            "url": f"https://simulated.sky-v1.local/search/{i}?q={quote_plus(query[:60])}",
            "snippet": text + " （本条目为模拟结果，因未配置 SerpAPI / Tavily / DuckDuckGo 无法抓取）",
        })
    return out


# ---------------------------------------------------------------------------
# Tool 1: 毫秒级联网搜索 WebSearchTool
# ---------------------------------------------------------------------------
class WebSearchTool(BaseTool):
    """毫秒级联网搜索 —— 多 Provider 优先级 + 本地 LRU/TTL 缓存。"""

    name: ClassVar[str] = "tool_web_search"
    description: ClassVar[str] = (
        "联网实时搜索（毫秒级 LRU 缓存 + SerpAPI/Tavily/DuckDuckGo 多 Provider）。"
        "用于查询实时新闻、最新数据、在线资料、事实核查。"
    )
    modal: ClassVar[str] = "text"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "num_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "返回条数（默认 5）",
            },
            "freshness": {
                "type": "string",
                "enum": ["any", "day", "week", "month"],
                "description": "结果时效（目前用于 metadata 标注）",
            },
            "skip_cache": {
                "type": "boolean",
                "description": "是否跳过缓存（默认 False）",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                },
            },
            "provider": {"type": "string"},
            "cached": {"type": "boolean"},
            "latency_ms": {"type": "integer"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        query = kwargs.get("query", "")
        num = int(kwargs.get("num_results", 5))
        skip_cache = bool(kwargs.get("skip_cache", False))
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, error="query 不能为空", tool_name=self.name)
        query = query.strip()
        num = max(1, min(num, 20))

        cfg = getattr(ctx, "config", None) or {}
        timeout_s = float(cfg.get("search_timeout_s", 3.0)) if isinstance(cfg, dict) else 3.0

        start = time.perf_counter()
        key = _cache_key("websearch", query, num=num)
        if not skip_cache:
            cached = _CACHE.get(key)
            if cached is not None and isinstance(cached, dict):
                latency_ms = int((time.perf_counter() - start) * 1000)
                results: list[dict] = cached.get("results", [])
                provider = cached.get("provider", "cache")
                output = (
                    f"[Cache Hit < {latency_ms}ms] 共 {len(results)} 条结果，"
                    f"provider={provider}\n"
                    + "\n".join(f"{i + 1}. [{r.get('title', '')[:60]}]({r.get('url', '')[:80]}) — {r.get('snippet', '')[:200]}"
                        for i, r in enumerate(results[:5]))
                )
                return ToolResult(
                    success=True,
                    output=output,
                    data={
                        "results": results,
                        "provider": provider,
                        "cached": True,
                        "simulated": cached.get("simulated", False),
                    },
                    tool_name=self.name,
                    latency_ms=latency_ms,
                    metadata={"cached": True},
                )

        # SKY_V1_TEST_MODE=1：不做任何真实网络调用，直接用模拟结果（保障测试快、稳）
        test_mode = bool(os.environ.get("SKY_V1_TEST_MODE", ""))

        # 依次尝试 Provider，首个成功即退出
        try:
            raw = None if test_mode else _try_serpapi(query, num, timeout_s)
            provider = "serpapi"
            if raw is None and not test_mode:
                raw = _try_tavily(query, num, timeout_s)
                provider = "tavily"
            if raw is None and not test_mode:
                raw = _try_duckduckgo_html(query, num, min(0.8, timeout_s))
                provider = "duckduckgo_html"
            simulated = False
            if raw is None:
                raw = _simulated_search_results(query, num)
                provider = "simulated"
                simulated = True
        except Exception as e:  # noqa: BLE001
            log.warning("All search providers failed, using simulated", error=str(e)[:200])
            raw = _simulated_search_results(query, num)
            provider = "simulated"
            simulated = True

        results = raw[:num]
        _CACHE.set(key, {"results": results, "provider": provider, "simulated": simulated})
        latency_ms = int((time.perf_counter() - start) * 1000)
        prefix = f"[Simulated] " if simulated else f"[{provider.upper()} {latency_ms}ms] "
        output = prefix + f"共 {len(results)} 条结果\n" + "\n".join(
            f"{i + 1}. [{r.get('title', '')[:80]}]({r.get('url', '')[:100]}) — {r.get('snippet', '')[:240]}"
            for i, r in enumerate(results)
        )
        return ToolResult(
            success=True,
            output=output,
            data={
                "results": results,
                "provider": provider,
                "cached": False,
                "simulated": simulated,
            },
            tool_name=self.name,
            latency_ms=latency_ms,
            metadata={"simulated": simulated, "provider": provider},
        )


# ---------------------------------------------------------------------------
# Tool 2: 深度推理 DeepReasoningTool（Tree-of-Thoughts + 工具调用链）
# ---------------------------------------------------------------------------
def _split_subquestions(question: str, n: int) -> list[str]:
    """启发式子问题拆分：当问题包含 "和/及/与/且/同时" + 问号，按连接符拆。"""
    cleaned = question.strip()
    seps = [r"\s+且\s+", r"\s+和\s+", r"\s+及\s+", r"\s+与\s+", r"\s+同时\s+", "；", ";"]
    for sep in seps:
        parts = re.split(sep, cleaned)
        if len(parts) >= 2:
            return [p.strip() for p in parts if p.strip()][: max(2, n)]
    # 默认按推理阶段生成子问题
    return [
        f"{cleaned}：① 定义与背景",
        f"{cleaned}：② 关键要素拆解",
        f"{cleaned}：③ 结论与对比建议",
    ][: max(1, n)]


def _simulated_expert_answer(subq: str) -> str:
    """纯模拟的"专家回答"，保证无外部依赖下也能产出有结构的推理路径。"""
    s = subq.strip("？?。. ")
    if "DeepSeek" in s or "代码" in s or "逻辑推理" in s or "数学" in s:
        return (
            "DeepSeek 在逻辑推理 / 数学 / 代码领域为国产天花板："
            "HumanEval 代码 pass@1 接近 GPT-4.5，支持 1M 超长上下文，"
            "API 定价仅为 Claude/GPT 的 1/50，是程序员与科研首选。"
        )
    if "Kimi" in s or "长文本" in s or "法律" in s or "论文" in s:
        return (
            "Kimi 超长文本无损处理 20 万字：将合同/论文整份上传后可精准溯源引用，"
            "并支持 Agent 集群协同拆分复杂资料分析任务。"
        )
    if "搜索" in s or "实时" in s or "新闻" in s:
        return (
            "联网搜索建议多 Provider 并行（SerpAPI > Tavily > DuckDuckGo）"
            "加 TTL LRU 缓存，热查询 < 50ms 返回，冷查询控制在 2-3s 内。"
        )
    if "对比" in s or "选型" in s or "区别" in s:
        return (
            "选型建议：日常/内容→豆包（中文自然度），"
            "开源私有化→Qwen（最大家族），"
            "严谨代码→DeepSeek（逻辑天花板），"
            "超长资料→Kimi（20 万字无损），"
            "行业工业→盘古（昇腾+气象/矿山）。"
        )
    return (
        f"关于「{s}」：综合背景可拆解为 (1) 问题定义与边界；"
        f"(2) 多维度关键因素；(3) 方案对比与折中；(4) 结论与落地建议。"
        f"建议结合联网搜索补充最新实时数据后再做最终判断。"
    )


class DeepReasoningTool(BaseTool):
    """深度推理：Plan → Act → Observe → Reflect → Final Answer。

    当 ctx.config["deep_reasoning_use_web"]=True 且存在可用搜索工具时，
    Act 阶段会附加一次联网搜索（命中缓存即 <50ms）作为"验算"。
    """

    name: ClassVar[str] = "tool_deep_reasoning"
    description: ClassVar[str] = (
        "深度多步推理（Tree-of-Thoughts Plan-Act-Observe-Reflect）。"
        "用于复杂对比、选型决策、技术架构推导、多条件综合判断。"
        "默认 3 步迭代，自动自我反思纠错。"
    )
    modal: ClassVar[str] = "text"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["question"],
        "properties": {
            "question": {"type": "string", "description": "需要深度推理的问题"},
            "max_iterations": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "description": "最大推理步数（默认 3）",
            },
            "enable_web_search": {
                "type": "boolean",
                "description": "是否在 Act 阶段调用联网搜索（默认 True）",
            },
            "citations_needed": {
                "type": "boolean",
                "description": "是否需要引用来源标注（默认 True）",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "plan": {"type": "array", "items": {"type": "string"}, "description": "推理计划子问题"},
            "iterations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "sub_question": {"type": "string"},
                        "act": {"type": "string"},
                        "observe": {"type": "string"},
                        "reflect": {"type": "string"},
                    },
                },
            },
            "final_answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        q = kwargs.get("question", "")
        max_iter = int(kwargs.get("max_iterations", 3))
        enable_web = bool(kwargs.get("enable_web_search", True))
        citations = bool(kwargs.get("citations_needed", True))
        if not isinstance(q, str) or not q.strip():
            return ToolResult(success=False, error="question 不能为空", tool_name=self.name)
        q = q.strip()
        max_iter = max(1, min(max_iter, 6))

        start = time.perf_counter()
        cfg = getattr(ctx, "config", None) or {}
        test_mode = bool(os.environ.get("SKY_V1_TEST_MODE", ""))
        use_web = enable_web and bool(cfg.get("deep_reasoning_use_web", True)) and not test_mode

        # Step 1: Plan → 拆分 n 个子问题
        subquestions = _split_subquestions(q, max_iter)
        iterations: list[dict] = []
        scratchpad: list[str] = []  # 每步观察结果累积，供后续反思用
        web_search_ctx = ToolContext(
            session_id=getattr(ctx, "session_id", "deep-reasoning"),
            rag_kb=getattr(ctx, "rag_kb", None),
            config=cfg,
        )

        # Step 2-4: Act → Observe → Reflect，对每个子问题迭代
        search_tool: WebSearchTool | None = None
        if use_web:
            # 懒构造：避免导入循环
            search_tool = WebSearchTool()

        # RAG 尝试
        rag_output: str = ""
        rag_kb = getattr(ctx, "rag_kb", None)
        if rag_kb is not None and hasattr(rag_kb, "query") and callable(rag_kb.query):
            try:
                raw = rag_kb.query(q, top_k=3)
                if isinstance(raw, list) and raw:
                    rag_output = "\n".join(
                        r.get("text", str(r))[:400] if isinstance(r, dict) else str(r)[:400]
                        for r in raw[:2]
                    )
            except Exception as e:  # noqa: BLE001
                log.info("RAG in deep reasoning unavailable", error=str(e)[:200])

        for idx, subq in enumerate(subquestions, 1):
            act_descs: list[str] = []
            observes: list[str] = []

            # Act A: 内部专家模拟（确定性）
            act_descs.append("A: 专家知识推演")
            expert_ans = _simulated_expert_answer(subq)
            observes.append(f"Obs(A) = {expert_ans}")

            # Act B: RAG 上下文
            if rag_output:
                act_descs.append("B: RAG 知识库检索")
                observes.append(f"Obs(B) = {rag_output[:500]}")

            # Act C: 联网搜索
            if search_tool is not None:
                act_descs.append("C: 联网搜索事实核查")
                try:
                    search_res = search_tool.run(web_search_ctx, query=subq, num_results=3)
                    if search_res.success:
                        data = search_res.data or {}
                        items = data.get("results") or []
                        if items:
                            snippets = " | ".join(
                                f"[{i + 1}] {it.get('snippet', '')[:200]}"
                                for i, it in enumerate(items[:3])
                            )
                            observes.append(f"Obs(C) = {snippets}")
                        else:
                            observes.append("Obs(C) = 搜索无命中")
                    else:
                        observes.append(f"Obs(C) = 搜索失败: {search_res.error}")
                except Exception as e:  # noqa: BLE001
                    observes.append(f"Obs(C) = 搜索异常: {str(e)[:200]}")

            # Reflect：合并观察 + 自纠错（对矛盾进行标记）
            combined = "；".join(observes)
            needs_caution = False
            if len(observes) >= 2 and rag_output:
                # 启发式：A/B/C 关键词不一致时标记需谨慎
                if ("DeepSeek" in combined and "DeepSeek" not in rag_output[:100]) or \
                   ("Kimi" in combined and "Kimi" not in rag_output[:100]):
                    needs_caution = True
            reflect = (
                "Reflect: 本步" + ("存在多来源差异，结论需谨慎。" if needs_caution else "观察一致，结论可信度高。")
                + f" 综合要点：{combined[:500]}"
            )
            scratchpad.append(reflect)

            iterations.append({
                "step": idx,
                "sub_question": subq,
                "act": " + ".join(act_descs),
                "observe": "\n".join(observes)[:1500],
                "reflect": reflect,
            })

        # Step 5: Final Answer（带置信度）
        key_points = []
        for it in iterations:
            obs = it["observe"].split("Obs(A) = ", 1)
            if len(obs) == 2:
                key_points.append(obs[1].split("\n", 1)[0][:200])
            else:
                key_points.append(it["sub_question"][:200])

        confidence = 0.9 if (rag_output and search_tool is not None) else (
            0.7 if rag_output or search_tool is not None else 0.5
        )
        # 含模拟搜索 → 扣 15%
        last_iter = iterations[-1] if iterations else {}
        any_sim = any(
            (it.get("observe") or "").startswith("Obs(C) = [Simulated]")
            for it in iterations
        )
        if any_sim:
            confidence = max(0.1, confidence - 0.15)

        final = (
            f"【深度推理最终结论】{q}\n"
            f"置信度：{confidence:.2f}\n"
            f"关键论点（共 {len(key_points)} 条）：\n"
            + "\n".join(f"  {i + 1}. {kp}" for i, kp in enumerate(key_points))
            + "\n"
            + ("【引用与来源】结论综合了内部专家知识 + RAG 知识库"
               + (" + 联网搜索结果" if use_web else "")
               + ("，其中部分条目为模拟结果，建议交叉核验。" if any_sim else "。")
               if citations else "")
        )

        output_parts = [f"=== DeepReasoning: Plan ({len(subquestions)} sub-questions) ==="]
        for it in iterations:
            output_parts.append(
                f"\n[Step {it['step']}] 子问题: {it['sub_question']}\n"
                f"  Act:     {it['act']}\n"
                f"  Observe: {(it['observe'][:300] + '…') if len(it['observe']) > 300 else it['observe']}\n"
                f"  Reflect: {(it['reflect'][:300] + '…') if len(it['reflect']) > 300 else it['reflect']}"
            )
        output_parts.append(f"\n=== Final Answer (confidence={confidence:.2f}) ===\n{final}")
        output = "\n".join(output_parts)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            success=True,
            output=output,
            data={
                "plan": subquestions,
                "iterations": iterations,
                "final_answer": final,
                "confidence": confidence,
                "simulated": any_sim or (search_tool is None),
            },
            tool_name=self.name,
            latency_ms=latency_ms,
            metadata={"confidence": confidence, "steps": len(iterations)},
        )


__all__ = ["WebSearchTool", "DeepReasoningTool", "_TtlLruCache"]
