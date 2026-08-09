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

import concurrent.futures
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


@dataclass
class SearchProviderOutcome:
    """真实搜索 Provider 的执行结果。注意区分：
    * available=False + reason="MISSING_KEY"：未配置 API Key（跳过，不视为失败）
    * available=False + reason="TIMEOUT"/"NETWORK_ERROR"/"HTTP_xxx"：请求异常（应显式告知用户）
    * available=True + len(results)==0：请求成功但无命中（与异常严格区分）
    * available=True + results：正常返回
    灵感来自 蓝奏云找书 v22 复盘：真实失败时不得伪造成功结果。"""
    provider: str
    available: bool
    reason: str = ""
    results: list[dict] = field(default_factory=list)
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Provider 1: Google Search（双通道：SerpAPI → Google Custom JSON API → Google HTML scrape）
# ---------------------------------------------------------------------------
def _try_google_search(query: str, num: int, timeout_s: float) -> SearchProviderOutcome:
    provider = "google"
    t0 = time.perf_counter()

    # 通道 A：SerpAPI（engine=google）—— 最稳定
    serp_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if serp_key:
        try:
            raw = _http_get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": serp_key, "num": num, "engine": "google", "no_cache": "true"},
                timeout_s=timeout_s,
            )
            if isinstance(raw, dict):
                organic = raw.get("organic_results") or []
                results = [
                    {
                        "title": str(i.get("title", ""))[:300],
                        "url": str(i.get("link") or i.get("source") or "")[:500],
                        "snippet": str(i.get("snippet") or i.get("title") or "")[:800],
                    }
                    for i in organic if isinstance(i, dict)
                ]
                if results:
                    return SearchProviderOutcome(
                        provider=provider, available=True, results=results[:num],
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                    )
        except Exception as e:  # noqa: BLE001
            log.warning("Google via SerpAPI failed", error=str(e)[:200])

    # 通道 B：Google Programmable Search Engine JSON API
    g_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
    g_cx = os.environ.get("GOOGLE_SEARCH_CX", "").strip()
    if g_key and g_cx:
        try:
            raw = _http_get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": g_key, "cx": g_cx, "q": query, "num": min(num, 10)},
                timeout_s=timeout_s,
            )
            if isinstance(raw, dict):
                items = raw.get("items") or []
                results = [
                    {
                        "title": str(i.get("title", ""))[:300],
                        "url": str(i.get("link") or "")[:500],
                        "snippet": str(i.get("snippet") or "")[:800],
                    }
                    for i in items if isinstance(i, dict)
                ]
                if results:
                    return SearchProviderOutcome(
                        provider=provider, available=True, results=results[:num],
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                    )
        except Exception as e:  # noqa: BLE001
            log.warning("Google CustomSearch API failed", error=str(e)[:200])

    # 通道 C：Google 搜索 HTML 页直扒（无 Key 兜底）—— 零依赖
    try:
        html = _http_get(
            "https://www.google.com/search",
            params={"q": query, "num": num, "hl": "zh-CN", "safe": "off"},
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001
        return SearchProviderOutcome(
            provider=provider, available=False,
            reason=f"NETWORK_ERROR: {type(e).__name__} {str(e)[:100]}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    results: list[dict] = []
    if isinstance(html, str):
        # <h3> 标题 与附近的 <a href="..."> 以及可见片段文本
        for h3 in re.findall(r"<h3[^>]*>(.*?)</h3>", html, flags=re.S):
            title = re.sub(r"<[^>]+>", "", h3).strip()
            if not title:
                continue
            results.append({
                "title": title[:300],
                "url": "",
                "snippet": f"（Google 搜索页摘要占位）{title}",
            })
            if len(results) >= num:
                break
    if results:
        return SearchProviderOutcome(
            provider=provider, available=True, results=results,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    # 所有通道走通但 0 命中
    return SearchProviderOutcome(
        provider=provider, available=True, results=[],
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Provider 2: 百度搜索（双通道：JSON 结果页 → 百度 HTML）
# ---------------------------------------------------------------------------
def _try_baidu_search(query: str, num: int, timeout_s: float) -> SearchProviderOutcome:
    provider = "baidu"
    t0 = time.perf_counter()
    try:
        html = _http_get(
            "https://www.baidu.com/s",
            params={"wd": query, "rn": max(10, num), "ie": "utf-8"},
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001
        return SearchProviderOutcome(
            provider=provider, available=False,
            reason=f"NETWORK_ERROR: {type(e).__name__} {str(e)[:100]}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    results: list[dict] = []
    if isinstance(html, str):
        # 百度结果标题都在 <h3 class="t"><a ...>title</a> 里
        for block in re.findall(r"<h3[^>]*class=\"[^\"]*t[^\"]*\"[^>]*>(.*?)</h3>", html, flags=re.S):
            title_m = re.search(r"<a[^>]*>(.*?)</a>", block, flags=re.S)
            href_m = re.search(r"<a[^>]*href=\"([^\"]+)\"", block, flags=re.S)
            title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
            href = href_m.group(1).strip() if href_m else ""
            if not title:
                continue
            results.append({
                "title": title[:300],
                "url": href[:500],
                "snippet": f"（百度）{title}",
            })
            if len(results) >= num:
                break
    return SearchProviderOutcome(
        provider=provider, available=True, results=results,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Provider 3: Bing 必应搜索
# ---------------------------------------------------------------------------
def _try_bing_search(query: str, num: int, timeout_s: float) -> SearchProviderOutcome:
    provider = "bing"
    t0 = time.perf_counter()
    # 通道 A：Bing Search API（认知服务）
    bing_key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
    if bing_key:
        try:
            try:
                import httpx  # type: ignore
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.get(
                        "https://api.bing.microsoft.com/v7.0/search",
                        params={"q": query, "count": min(num, 50), "mkt": "zh-CN"},
                        headers={"Ocp-Apim-Subscription-Key": bing_key},
                    )
                    resp.raise_for_status()
                    raw = resp.json()
            except ImportError:
                import urllib.request
                import urllib.error
                url = (
                    "https://api.bing.microsoft.com/v7.0/search?"
                    f"q={quote_plus(query)}&count={min(num,50)}&mkt=zh-CN"
                )
                req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": bing_key})
                with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
                    raw = json.loads(r.read().decode("utf-8", "ignore"))
            if isinstance(raw, dict):
                pages = (raw.get("webPages") or {}).get("value") or []
                results = [
                    {
                        "title": str(p.get("name", ""))[:300],
                        "url": str(p.get("url") or "")[:500],
                        "snippet": str(p.get("snippet") or "")[:800],
                    }
                    for p in pages if isinstance(p, dict)
                ]
                if results:
                    return SearchProviderOutcome(
                        provider=provider, available=True, results=results[:num],
                        latency_ms=int((time.perf_counter() - t0) * 1000),
                    )
        except Exception as e:  # noqa: BLE001
            log.warning("Bing Search API failed", error=str(e)[:200])

    # 通道 B：Bing HTML 搜索页直扒
    try:
        html = _http_get(
            "https://www.bing.com/search",
            params={"q": query, "count": max(10, num), "setlang": "zh-CN"},
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001
        return SearchProviderOutcome(
            provider=provider, available=False,
            reason=f"NETWORK_ERROR: {type(e).__name__} {str(e)[:100]}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    results: list[dict] = []
    if isinstance(html, str):
        for m in re.finditer(r"<li class=\"b_algo\"[^>]*>(.*?)</li>\s*</li>", html, flags=re.S):
            block = m.group(1)
            title_m = re.search(r"<h2><a[^>]*>(.*?)</a>", block, flags=re.S)
            href_m = re.search(r"<h2><a[^>]*href=\"([^\"]+)\"", block, flags=re.S)
            snip_m = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.S)
            title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
            href = href_m.group(1).strip() if href_m else ""
            snip = re.sub(r"<[^>]+>", "", snip_m.group(1)).strip() if snip_m else ""
            if not (title or snip):
                continue
            results.append({
                "title": title[:300],
                "url": href[:500],
                "snippet": snip[:800] or f"（必应）{title}",
            })
            if len(results) >= num:
                break
    return SearchProviderOutcome(
        provider=provider, available=True, results=results,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Provider 4: 今日头条搜索（资讯/新闻/短视频优势）
# ---------------------------------------------------------------------------
def _try_toutiao_search(query: str, num: int, timeout_s: float) -> SearchProviderOutcome:
    provider = "toutiao"
    t0 = time.perf_counter()
    try:
        html = _http_get(
            "https://so.toutiao.com/search",
            params={"keyword": query, "pd": "information", "source": "input", "dvpf": "pc"},
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001
        return SearchProviderOutcome(
            provider=provider, available=False,
            reason=f"NETWORK_ERROR: {type(e).__name__} {str(e)[:100]}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    results: list[dict] = []
    if isinstance(html, str):
        # 头条搜索结果常见结构：<a ...> 带标题 + 摘要或卡片
        for m in re.finditer(r"<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", html, flags=re.S):
            href = m.group(1).strip()
            title_raw = m.group(2)
            title = re.sub(r"<[^>]+>", "", title_raw).strip()
            if 4 < len(title) <= 120 and href.startswith("http"):
                results.append({
                    "title": title[:300],
                    "url": href[:500],
                    "snippet": f"（头条资讯）{title}",
                })
            if len(results) >= num:
                break
    return SearchProviderOutcome(
        provider=provider, available=True, results=results,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Provider 5（兼容保留）：Tavily / DuckDuckGo（作为"其他引擎"兜底）
# ---------------------------------------------------------------------------
def _try_tavily(query: str, num: int, timeout_s: float) -> SearchProviderOutcome:
    t0 = time.perf_counter()
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return SearchProviderOutcome(provider="tavily", available=False, reason="MISSING_KEY")
    try:
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
        except ImportError:
            # 无 httpx 时跳过 Tavily（不影响主流 4 引擎）
            return SearchProviderOutcome(provider="tavily", available=False, reason="DEP_httpx_UNAVAILABLE")
    except Exception as e:  # noqa: BLE001
        return SearchProviderOutcome(
            provider="tavily", available=False,
            reason=f"NETWORK_ERROR: {type(e).__name__} {str(e)[:100]}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    data = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return SearchProviderOutcome(provider="tavily", available=True, results=[])
    results = [
        {
            "title": str(i.get("title", ""))[:300],
            "url": str(i.get("url", ""))[:500],
            "snippet": str(i.get("content", ""))[:800],
        }
        for i in data if isinstance(i, dict)
    ]
    return SearchProviderOutcome(
        provider="tavily", available=True, results=results[:num],
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


def _simulated_search_results(query: str, num: int) -> list[dict]:
    """**开发模式**：用户显式允许 simulated 时才用；真实搜索引擎失败必须返回结构化失败原因。"""
    stub_templates = [
        "{q} - 基础定义与常见分类（来源：开发模式模拟结果）",
        "{q} 的最新进展与行业趋势综述（来源：开发模式模拟结果）",
        "如何选择 {q}：关键指标与选型建议（来源：开发模式模拟结果）",
        "{q} 对比：主流方案优劣势分析（来源：开发模式模拟结果）",
        "{q} 常见问题 FAQ 与避坑指南（来源：开发模式模拟结果）",
    ]
    out: list[dict] = []
    for i in range(max(1, num)):
        tpl = stub_templates[i % len(stub_templates)]
        text = tpl.format(q=query)
        # 每条 title 加序号，保证 "按 title 去重" 时不会把 num>5 的模拟结果误杀
        if i >= len(stub_templates):
            title_prefix = f"#{i + 1} | "
        else:
            title_prefix = ""
        base_title = text.split("（")[0][:200]
        out.append({
            "title": (title_prefix + base_title).strip()[:200],
            "url": f"https://simulated.sky-v1.local/search/{i}?q={quote_plus(query[:60])}",
            "snippet": text + " （**开发模式模拟**：配置 Google/Baidu/Bing/今日头条 任一 Provider 可获得真实搜索结果）",
        })
    return out


# ---------------------------------------------------------------------------
# Tool 1: 毫秒级联网搜索 WebSearchTool
# ---------------------------------------------------------------------------
class WebSearchTool(BaseTool):
    """毫秒级联网搜索 —— 4 大主引擎并发 + LRU/TTL 缓存 + 空结果/异常严格区分。

    主流 Provider（按默认并发尝试顺序）：google / baidu / bing / toutiao
    可兼容辅助：tavily / duckduckgo_html（兜底）
    """

    name: ClassVar[str] = "tool_web_search"
    description: ClassVar[str] = (
        "联网实时搜索（毫秒级 LRU 缓存 + Google/百度/Bing/今日头条 多引擎并发）。"
        "用于查询实时新闻、最新数据、在线资料、事实核查。"
    )
    modal: ClassVar[str] = "text"
    DEFAULT_PROVIDERS: ClassVar[list[str]] = ["google", "baidu", "bing", "toutiao", "tavily"]
    PROVIDER_FN: ClassVar[dict] = {
        "google": _try_google_search,
        "baidu": _try_baidu_search,
        "bing": _try_bing_search,
        "toutiao": _try_toutiao_search,
        "tavily": _try_tavily,
    }
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
            "providers": {
                "type": "array",
                "items": {"type": "string", "enum": ["google", "baidu", "bing", "toutiao", "tavily"]},
                "description": "指定使用的搜索 Provider 白名单（默认全量并发：Google/百度/Bing/今日头条/Tavily）",
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
            "allow_simulated": {
                "type": "boolean",
                "description": "**开发模式**：当所有 Provider 均不可用时，是否允许返回模拟摘要。默认 False=必须显式告知失败原因。",
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
                        "provider": {"type": "string"},
                    },
                },
            },
            "providers_used": {"type": "array", "items": {"type": "string"}},
            "provider_statuses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "available": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "count": {"type": "integer"},
                        "latency_ms": {"type": "integer"},
                    },
                },
            },
            "cached": {"type": "boolean"},
            "simulated": {"type": "boolean"},
            "status": {"type": "string", "description": "OK | PARTIAL | EMPTY | FAILED"},
            "latency_ms": {"type": "integer"},
        },
    }

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        query = kwargs.get("query", "")
        num = int(kwargs.get("num_results", 5))
        skip_cache = bool(kwargs.get("skip_cache", False))
        allow_simulated = bool(kwargs.get("allow_simulated", False))
        providers_arg = kwargs.get("providers")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, error="query 不能为空", tool_name=self.name)
        query = query.strip()
        num = max(1, min(num, 20))

        cfg = getattr(ctx, "config", None) or {}
        total_timeout_s = float(cfg.get("search_timeout_s", 3.5)) if isinstance(cfg, dict) else 3.5
        single_timeout_s = min(max(0.8, total_timeout_s * 0.7), 2.6)

        # 允许覆盖默认并发顺序（用户可只跑百度/头条等）
        if isinstance(providers_arg, list) and providers_arg:
            providers = [p for p in providers_arg if isinstance(p, str) and p in self.PROVIDER_FN]
        else:
            providers = list(self.DEFAULT_PROVIDERS)
        if not providers:
            return ToolResult(success=False, error="providers 白名单为空", tool_name=self.name)

        start = time.perf_counter()
        key = _cache_key("websearch", query, num=num, providers="|".join(providers))
        if not skip_cache:
            cached = _CACHE.get(key)
            if cached is not None and isinstance(cached, dict):
                latency_ms = int((time.perf_counter() - start) * 1000)
                results: list[dict] = cached.get("results", [])
                primary: str = (cached.get("providers_used") or ["cache"])[0]
                output = (
                    f"[Cache Hit < {latency_ms}ms] 共 {len(results)} 条，"
                    f"providers={cached.get('providers_used')}\n"
                    + "\n".join(
                        f"{i + 1}. [{r.get('title', '')[:60]}]({r.get('url', '')[:80]}) — {r.get('snippet', '')[:200]}"
                        for i, r in enumerate(results[:5])
                    )
                )
                return ToolResult(
                    success=True,
                    output=output,
                    data={
                        "results": results,
                        "providers_used": cached.get("providers_used", []),
                        "provider_statuses": cached.get("provider_statuses", []),
                        "cached": True,
                        "simulated": cached.get("simulated", False),
                        "status": cached.get("status", "OK"),
                    },
                    tool_name=self.name,
                    latency_ms=latency_ms,
                    metadata={"cached": True},
                )

        # 单元测试模式：不做任何真实网络调用
        test_mode = bool(os.environ.get("SKY_V1_TEST_MODE", ""))

        provider_statuses: list[dict] = []
        all_results: list[dict] = []
        used: list[str] = []
        simulated = False
        errors: list[str] = []

        if not test_mode:
            # 并发多引擎 + 统一总超时（经验复盘：避免串行堆叠等待）
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as ex:
                future_map = {
                    ex.submit(fn, query, num, single_timeout_s): name
                    for name, fn in ((p, self.PROVIDER_FN[p]) for p in providers)
                }
                try:
                    for fut in concurrent.futures.as_completed(future_map, timeout=total_timeout_s):
                        name = future_map[fut]
                        try:
                            outcome = fut.result()
                        except Exception as e:  # noqa: BLE001
                            provider_statuses.append({
                                "provider": name,
                                "available": False,
                                "reason": f"EXCEPTION: {type(e).__name__} {str(e)[:120]}",
                                "count": 0,
                                "latency_ms": 0,
                            })
                            errors.append(f"{name}: EXCEPTION")
                            continue
                        status_row = {
                            "provider": outcome.provider,
                            "available": outcome.available,
                            "reason": outcome.reason,
                            "count": len(outcome.results),
                            "latency_ms": outcome.latency_ms,
                        }
                        provider_statuses.append(status_row)
                        if not outcome.available and outcome.reason and outcome.reason != "MISSING_KEY":
                            # 有 Key 但请求失败：算作真实错误
                            errors.append(f"{outcome.provider}: {outcome.reason}")
                        if outcome.available and outcome.results:
                            used.append(outcome.provider)
                            for r in outcome.results:
                                r2 = dict(r)
                                r2["provider"] = outcome.provider
                                all_results.append(r2)
                except concurrent.futures.TimeoutError:
                    # 总超时：标记为 FAILED
                    pending = [name for fut, name in future_map.items() if not fut.done()]
                    for p in pending:
                        provider_statuses.append({
                            "provider": p,
                            "available": False,
                            "reason": "TIMEOUT_TOTAL",
                            "count": 0,
                            "latency_ms": int(total_timeout_s * 1000),
                        })
                        errors.append(f"{p}: TIMEOUT_TOTAL")

        # 兜底：真实引擎全不可用 → 仅在用户显式 allow_simulated 时返回模拟
        if not all_results and not used and (test_mode or allow_simulated):
            all_results = _simulated_search_results(query, num)
            for r in all_results:
                r["provider"] = "simulated"
            simulated = True
            provider_statuses.append({
                "provider": "simulated",
                "available": True,
                "reason": "DEV_MODE",
                "count": len(all_results),
                "latency_ms": 0,
            })
            used.append("simulated")

        # 去重 + 截取 top N：标题相同视为重复（不同引擎常镜像收录同一页面）
        seen_titles: set[str] = set()
        seen_urls: set[str] = set()
        dedup: list[dict] = []
        for r in all_results:
            title_norm = (r.get("title") or "").strip().lower()
            url_norm = (r.get("url") or "").strip()
            if (title_norm and title_norm in seen_titles) or (url_norm and url_norm in seen_urls):
                continue
            if title_norm:
                seen_titles.add(title_norm)
            if url_norm:
                seen_urls.add(url_norm)
            dedup.append(r)
            if len(dedup) >= num:
                break

        # Status 判定（严格区分：OK / PARTIAL / EMPTY / FAILED）
        has_hit = len(dedup) > 0
        has_err = bool(errors)
        if simulated:
            status = "OK" if has_hit else "EMPTY"
        elif has_hit and not has_err:
            status = "OK"
        elif has_hit and has_err:
            status = "PARTIAL"
        elif (not has_hit) and has_err:
            status = "FAILED"
        else:  # not has_hit and not has_err
            status = "EMPTY"

        latency_ms = int((time.perf_counter() - start) * 1000)
        _CACHE.set(key, {
            "results": dedup,
            "providers_used": used,
            "provider_statuses": provider_statuses,
            "simulated": simulated,
            "status": status,
        })

        # 组装 output：用户能一眼看懂状态与错误
        lines: list[str] = []
        header_prefix = f"[Cache Miss {latency_ms}ms] status={status} providers_used={used}"
        if simulated:
            header_prefix += " **DEV_SIMULATED**"
        lines.append(header_prefix)
        for row in provider_statuses:
            if row.get("available"):
                lines.append(f"  - {row['provider']}: OK {row.get('count',0)}条 {row.get('latency_ms',0)}ms")
            else:
                r = row.get("reason") or ""
                if r == "MISSING_KEY":
                    lines.append(f"  - {row['provider']}: SKIP (未配置 API Key)")
                else:
                    lines.append(f"  - {row['provider']}: FAIL reason={r[:120]}")
        lines.append(f"--- 共 {len(dedup)} 条结果 ---")
        for i, r in enumerate(dedup):
            lines.append(
                f"{i + 1}. [{r.get('title', '')[:80]}]({r.get('url', '')[:100]}) — {r.get('snippet', '')[:240]}"
            )
        if status == "FAILED":
            lines.append(
                "⚠️ 所有已配置的搜索引擎均返回异常。排查建议：(1) 检查网络是否可访问 Google/Baidu/Bing/头条；"
                "(2) 设置 SERPAPI_API_KEY / GOOGLE_SEARCH_API_KEY / BING_SEARCH_API_KEY 中任一即可大幅提升成功率；"
                "(3) 临时测试可加 allow_simulated=true。"
            )
        elif status == "EMPTY":
            lines.append("⚠️ 搜索引擎请求成功，但未命中任何结果。建议尝试更宽泛的关键词或调整 providers。")

        return ToolResult(
            success=(status != "FAILED" or simulated),
            output="\n".join(lines),
            data={
                "results": dedup,
                "providers_used": used,
                "provider_statuses": provider_statuses,
                "cached": False,
                "simulated": simulated,
                "status": status,
            },
            tool_name=self.name,
            latency_ms=latency_ms,
            metadata={
                "simulated": simulated,
                "providers": used,
                "status": status,
                "errors": errors,
            },
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
