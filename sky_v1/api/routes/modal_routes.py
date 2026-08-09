from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Request

from sky_v1.api.types import (
    AudioSpeechRequest,
    AudioSpeechResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    ThreeDGenerationRequest,
    ThreeDGenerationResponse,
    ThreeDObject,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoObject,
)
from sky_v1.agent.base import ToolContext

router = APIRouter()


def _sim_url(prefix: str, suffix: str, key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"https://sim.sky-v1.local/{prefix}/{h}.{suffix}"


def _call_tool_safe(agent: Any, tool_name: str, **kwargs) -> Any:
    if agent is None:
        return None
    try:
        tool_attr = getattr(agent, tool_name, None)
        if tool_attr is None:
            return None
        if callable(tool_attr):
            return tool_attr(**kwargs)
    except Exception:
        return None
    return None


def _make_ctx(request: Request) -> ToolContext:
    """构造带 config 的 ToolContext，供需要 API Key 的工具使用。"""
    rag_kb = getattr(request.app.state, "rag_kb", None)
    cfg = getattr(request.app.state, "config", None) or {}
    session_id = request.headers.get("x-sky-session-id", "api-call")
    user_id = request.headers.get("x-sky-user-id", "anonymous")
    return ToolContext(session_id=session_id, user_id=user_id, rag_kb=rag_kb, config=cfg)


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def images_generations(
    request: Request,
    body: ImageGenerationRequest,
) -> ImageGenerationResponse:
    agent = getattr(request.app.state, "agent", None)
    engine = getattr(request.app.state, "engine", None)
    data: list[ImageObject] = []
    for i in range(body.n):
        url = ""
        revised = body.prompt
        # 优先走 SkyInferenceEngine（真实模型生成）
        if engine is not None:
            try:
                result = engine.generate(modality="image", prompt=body.prompt) if hasattr(engine, "generate") else None
                if isinstance(result, dict):
                    tensor = result.get("tensor")
                    shape = result.get("shape", [])
                    if tensor is not None or shape:
                        url = f"data:application/x-sky-v1-tensor;shape={','.join(str(s) for s in shape)};base64,{_sim_url('img','png',body.prompt)[:24]}"
                        revised = f"[Engine] image tensor shape={shape}"
            except Exception:
                pass
        if not url:
            # Fallback: Agent 工具
            result = _call_tool_safe(agent, "tool_image_generate", prompt=body.prompt, size=body.size)
            if isinstance(result, dict):
                url = str(result.get("url", ""))
                revised = str(result.get("revised_prompt", body.prompt))
        if not url:
            url = _sim_url("img", "png", f"{body.prompt}:{body.size}:{i}")
        data.append(ImageObject(url=url, revised_prompt=revised))
    return ImageGenerationResponse(data=data)


@router.post("/audio/speech", response_model=AudioSpeechResponse)
async def audio_speech(
    request: Request,
    body: AudioSpeechRequest,
) -> AudioSpeechResponse:
    """TTS：豆包语音合成大模型 2.0（seed-tts-2.0），未配置时回退模拟。"""
    url = ""
    duration_ms = 0
    try:
        from sky_v1.agent.tools.audio_tools import TTSTool

        ctx = _make_ctx(request)
        result = TTSTool().run(
            ctx,
            text=body.input,
            voice=body.voice,
            response_format=body.response_format,
            speed=body.speed,
        )
        if result.success and result.data:
            url = str(result.data.get("audio_url", ""))
            duration_ms = int(result.data.get("duration_ms", 0))
    except Exception:
        pass
    if not url:
        url = _sim_url("audio", body.response_format, f"{body.voice}:{body.input}")
        duration_ms = max(100, len(body.input) * 80)
    return AudioSpeechResponse(url=url, duration_ms=duration_ms)


@router.post("/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def audio_transcriptions(
    request: Request,
    body: AudioTranscriptionRequest,
) -> AudioTranscriptionResponse:
    """ASR：豆包录音文件识别极速版（volc.bigasr.auc_turbo），未配置时回退模拟。"""
    text = ""
    segments: list[dict[str, Any]] = []
    try:
        from sky_v1.agent.tools.audio_tools import ASRTool

        ctx = _make_ctx(request)
        result = ASRTool().run(
            ctx,
            audio_url=body.file_url,
            language=body.language,
        )
        if result.success and result.data:
            text = str(result.data.get("text", "") or result.output or "")
            segs = result.data.get("segments", [])
            if isinstance(segs, list):
                segments = [s if isinstance(s, dict) else {"text": str(s)} for s in segs]
    except Exception:
        pass
    if not text:
        text = f"[SIM ASR] 音频转录结果（基于URL哈希：{body.file_url[-16:]}）"
        segments = [{"start": 0, "end": 3, "text": text}]
    return AudioTranscriptionResponse(text=text, segments=segments)


@router.post("/videos/generations", response_model=VideoGenerationResponse)
async def videos_generations(
    request: Request,
    body: VideoGenerationRequest,
) -> VideoGenerationResponse:
    agent = getattr(request.app.state, "agent", None)
    data: list[VideoObject] = []
    for i in range(body.n):
        url = ""
        duration_s = body.duration_s
        result = _call_tool_safe(
            agent,
            "tool_video_generate",
            prompt=body.prompt,
            duration_s=body.duration_s,
        )
        if isinstance(result, dict):
            url = str(result.get("url", ""))
            duration_s = int(result.get("duration_s", body.duration_s))
        if not url:
            url = _sim_url("video", "mp4", f"{body.prompt}:{body.duration_s}:{i}")
        data.append(VideoObject(url=url, duration_s=duration_s))
    return VideoGenerationResponse(data=data)


@router.post("/3d/generations", response_model=ThreeDGenerationResponse)
async def three_d_generations(
    request: Request,
    body: ThreeDGenerationRequest,
) -> ThreeDGenerationResponse:
    agent = getattr(request.app.state, "agent", None)
    data: list[ThreeDObject] = []
    fmt = body.format_ or "glb"
    for i in range(body.n):
        url = ""
        result_fmt = fmt
        result = _call_tool_safe(
            agent,
            "tool_3d_mesh",
            prompt=body.prompt,
            format=fmt,
        )
        if isinstance(result, dict):
            url = str(result.get("url", ""))
            result_fmt = str(result.get("format", fmt))
        if not url:
            url = _sim_url("3d", fmt, f"{body.prompt}:{fmt}:{i}")
        data.append(ThreeDObject(url=url, format=result_fmt))
    return ThreeDGenerationResponse(data=data)
