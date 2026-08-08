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


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def images_generations(
    request: Request,
    body: ImageGenerationRequest,
) -> ImageGenerationResponse:
    agent = getattr(request.app.state, "agent", None)
    data: list[ImageObject] = []
    for i in range(body.n):
        url = ""
        revised = body.prompt
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
    agent = getattr(request.app.state, "agent", None)
    url = ""
    duration_ms = 0
    result = _call_tool_safe(
        agent,
        "tool_tts",
        text=body.input,
        voice=body.voice,
        response_format=body.response_format,
        speed=body.speed,
    )
    if isinstance(result, dict):
        url = str(result.get("url", ""))
        duration_ms = int(result.get("duration_ms", 0))
    if not url:
        url = _sim_url("audio", body.response_format, f"{body.voice}:{body.input}")
        duration_ms = max(100, len(body.input) * 80)
    return AudioSpeechResponse(url=url, duration_ms=duration_ms)


@router.post("/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def audio_transcriptions(
    request: Request,
    body: AudioTranscriptionRequest,
) -> AudioTranscriptionResponse:
    agent = getattr(request.app.state, "agent", None)
    text = ""
    segments: list[dict[str, Any]] = []
    result = _call_tool_safe(
        agent,
        "tool_asr",
        file_url=body.file_url,
        language=body.language,
    )
    if isinstance(result, dict):
        text = str(result.get("text", ""))
        segs = result.get("segments", [])
        if isinstance(segs, list):
            segments = [s if isinstance(s, dict) else {"text": str(s)} for s in segs]
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
