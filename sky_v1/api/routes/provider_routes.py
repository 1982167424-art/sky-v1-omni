"""Provider 元数据 + 透传转发路由（终端网页版）。

提供：
  - GET  /v1/providers        列出所有 LLM Provider 元数据（与 Electron PROVIDER_META 一致）
  - POST /v1/providers/chat   透传转发到外部厂商（OpenAI 兼容协议，支持 SSE 流式）

透传转发通过请求头传入凭据（后端不持久化 API Key）：
  X-Provider-Base-URL : 厂商 base_url（如 https://api.minimax.io/v1）
  X-Provider-API-Key  : 厂商 API Key（Bearer Token）
  X-Provider-Extra    : 额外 header（JSON，如 {"anthropic-version":"2023-06-01"}）
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from sky_v1.utils.logging import get_logger

log = get_logger("api.routes.provider")

router = APIRouter()

# Provider 元数据（与 desktop/src/main/index.ts 的 PROVIDER_META 保持一致）
PROVIDERS: list[dict[str, Any]] = [
    {
        "id": "volcengine",
        "name": "火山引擎方舟 (Ark)",
        "brandColor": "#1664FF",
        "defaultBaseUrl": "https://ark.cn-beijing.volces.com/api/v3",
        "docsUrl": "https://www.volcengine.com/docs/82379",
        "modelLabel": "推理接入点 Endpoint ID",
        "modelPlaceholder": "ep-2024xxxxxxxxxxxxxxxx",
        "supportsStreaming": True,
        "defaultModel": "",
    },
    {
        "id": "minimax",
        "name": "MiniMax (M3)",
        "brandColor": "#FF6900",
        "defaultBaseUrl": "https://api.minimax.io/v1",
        "docsUrl": "https://platform.minimaxi.com/docs",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 MiniMax-M3",
        "supportsStreaming": True,
        "defaultModel": "MiniMax-M3",
    },
    {
        "id": "dashscope",
        "name": "阿里云 通义千问 (DashScope)",
        "brandColor": "#FF6A00",
        "defaultBaseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "docsUrl": "https://help.aliyun.com/zh/dashscope",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 qwen-plus / qwen-turbo / qwen-max",
        "supportsStreaming": True,
        "defaultModel": "qwen-plus",
    },
    {
        "id": "zhipu",
        "name": "智谱 AI (GLM / 清言)",
        "brandColor": "#0080FF",
        "defaultBaseUrl": "https://open.bigmodel.cn/api/paas/v4",
        "docsUrl": "https://open.bigmodel.cn/dev/api",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 glm-5.1 / glm-4.5-flash / glm-4.6",
        "supportsStreaming": True,
        "defaultModel": "glm-5.1",
    },
    {
        "id": "moonshot",
        "name": "月之暗面 Moonshot (Kimi)",
        "brandColor": "#3370FF",
        "defaultBaseUrl": "https://api.moonshot.cn/v1",
        "docsUrl": "https://platform.moonshot.cn/docs",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 kimi-k2-0905 / moonshot-v1-128k",
        "supportsStreaming": True,
        "defaultModel": "kimi-k2-0905",
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "brandColor": "#4143B3",
        "defaultBaseUrl": "https://api.deepseek.com/v1",
        "docsUrl": "https://api-docs.deepseek.com",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 deepseek-chat / deepseek-reasoner",
        "supportsStreaming": True,
        "defaultModel": "deepseek-chat",
    },
    {
        "id": "nvidia",
        "name": "NVIDIA NIM",
        "brandColor": "#76b900",
        "defaultBaseUrl": "https://integrate.api.nvidia.com/v1",
        "docsUrl": "https://docs.api.nvidia.com",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 moonshotai/kimi-k2.6 / deepseek-ai/deepseek-r1",
        "supportsStreaming": True,
        "defaultModel": "moonshotai/kimi-k2.6",
    },
    {
        "id": "mimo",
        "name": "小米 MiMo",
        "brandColor": "#FF7043",
        "defaultBaseUrl": "https://api.xiaomimimo.com/v1",
        "docsUrl": "https://platform.xiaomimimo.com/docs/zh-CN",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 mimo-v2.5-pro / mimo-v2-flash",
        "supportsStreaming": True,
        "defaultModel": "mimo-v2.5-pro",
    },
    {
        "id": "longcat",
        "name": "美团 LongCat (龙猫)",
        "brandColor": "#FFD100",
        "defaultBaseUrl": "https://api.longcat.chat/openai/v1",
        "docsUrl": "https://longcat.chat/platform/docs/zh",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 LongCat-Flash-Chat / LongCat-Flash-Thinking",
        "supportsStreaming": True,
        "defaultModel": "LongCat-Flash-Chat",
    },
    {
        "id": "anthropic",
        "name": "Anthropic Claude",
        "brandColor": "#D97757",
        "defaultBaseUrl": "https://api.anthropic.com/v1",
        "docsUrl": "https://docs.anthropic.com/en/api",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 claude-sonnet-4-5-20250929 / claude-opus-4-1",
        "supportsStreaming": True,
        "defaultModel": "claude-sonnet-4-5-20250929",
        "extraHeaders": {"anthropic-version": "2023-06-01"},
    },
    {
        "id": "google",
        "name": "Google Gemini",
        "brandColor": "#4285F4",
        "defaultBaseUrl": "https://generativelanguage.googleapis.com/v1beta/openai",
        "docsUrl": "https://ai.google.dev/gemini-api/docs/openai",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 gemini-2.5-pro / gemini-2.5-flash",
        "supportsStreaming": True,
        "defaultModel": "gemini-2.5-flash",
    },
    {
        "id": "openai",
        "name": "OpenAI (GPT)",
        "brandColor": "#10a37f",
        "defaultBaseUrl": "https://api.openai.com/v1",
        "docsUrl": "https://platform.openai.com/docs",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 gpt-5 / gpt-4o / o3",
        "supportsStreaming": True,
        "defaultModel": "gpt-5",
    },
    {
        "id": "sensetime",
        "name": "商汤 SenseNova (日日新)",
        "brandColor": "#00C2B2",
        "defaultBaseUrl": "https://token.sensenova.cn/v1",
        "docsUrl": "https://platform.sensenova.cn/docs",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 sensenova-6.7-flash-lite / deepseek-v4-flash",
        "supportsStreaming": True,
        "defaultModel": "sensenova-6.7-flash-lite",
    },
    {
        "id": "siliconflow",
        "name": "硅基流动 SiliconFlow",
        "brandColor": "#FF6A00",
        "defaultBaseUrl": "https://api.siliconflow.cn/v1",
        "docsUrl": "https://docs.siliconflow.cn/cn/api-reference/chat-completions/",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 deepseek-ai/DeepSeek-V3.2 / zai-org/GLM-5.1",
        "supportsStreaming": True,
        "defaultModel": "deepseek-ai/DeepSeek-V3.2",
    },
    {
        "id": "telechat",
        "name": "中国电信 星辰 TeleChat",
        "brandColor": "#0066CC",
        "defaultBaseUrl": "https://ai.ctaigw.cn/v1",
        "docsUrl": "https://www.teleai.com.cn",
        "modelLabel": "模型 ID",
        "modelPlaceholder": "例如 TeleChat3-105B-A4.7-Thinking / TeleChat3-36B-Thinking",
        "supportsStreaming": True,
        "defaultModel": "TeleChat3-105B-A4.7-Thinking",
    },
    {
        "id": "openai-compatible",
        "name": "OpenAI 兼容（自定义 Base URL）",
        "brandColor": "#10a37f",
        "defaultBaseUrl": "https://api.openai.com/v1",
        "docsUrl": "https://platform.openai.com/docs",
        "modelLabel": "模型 / 部署名",
        "modelPlaceholder": "自定义模型名",
        "supportsStreaming": True,
        "defaultModel": "gpt-4o-mini",
    },
]


@router.get("/providers")
async def list_providers() -> dict[str, Any]:
    """列出所有支持的 LLM Provider 元数据（终端网页版 UI 渲染用）。"""
    return {"providers": PROVIDERS, "count": len(PROVIDERS)}


@router.post("/providers/chat")
async def providers_chat(
    request: Request,
    x_provider_base_url: str = Header("", alias="X-Provider-Base-URL"),
    x_provider_api_key: str = Header("", alias="X-Provider-API-Key"),
    x_provider_extra: str = Header("", alias="X-Provider-Extra"),
) -> Any:
    """透传转发到外部厂商（OpenAI 兼容 /chat/completions，支持 SSE 流式）。

    后端不持久化 API Key，凭据由调用方通过请求头传入。
    """
    if not x_provider_base_url or not x_provider_api_key:
        return {
            "error": "Missing X-Provider-Base-URL or X-Provider-API-Key header",
            "hint": "终端网页版调用外部厂商需在请求头传入凭据，后端不存储 API Key",
        }

    body = await request.json()
    base_url = x_provider_base_url.rstrip("/")
    url = f"{base_url}/chat/completions"

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {x_provider_api_key}",
    }
    if x_provider_extra:
        try:
            extra = json.loads(x_provider_extra)
            if isinstance(extra, dict):
                headers.update({str(k): str(v) for k, v in extra.items()})
        except Exception:
            pass

    stream = body.get("stream", False)

    if not stream:
        # 非流式：直接转发并返回 JSON
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, headers=headers, json=body)
                return resp.json()
        except httpx.HTTPError as e:
            log.warning("providers_chat non-stream error", error=str(e))
            return {"error": f"Upstream request failed: {str(e)}"}

    # 流式：SSE 透传
    async def sse_generator():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.HTTPError as e:
            err = f"data: {json.dumps({'error': str(e)})}\n\n"
            yield err.encode("utf-8")

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
