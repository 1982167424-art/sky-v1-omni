"""Audio-family tools: ASRTool (speech->text), TTSTool (text->speech)."""
from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.audio")


def _md5(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"_md5 expects str, got {type(s).__name__}")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class ASRTool(BaseTool):
    name: ClassVar[str] = "tool_asr"
    description: ClassVar[str] = "Automatic Speech Recognition: audio URL -> transcript."
    modal: ClassVar[str] = "audio"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["audio_url"],
        "properties": {
            "audio_url": {"type": "string", "description": "Audio URL to transcribe"},
            "language": {
                "type": "string",
                "description": "Language hint (zh, en, auto)",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "Transcript text"},
            "segments": {"type": "array"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        audio_url = kwargs.get("audio_url", "")
        language = kwargs.get("language", "zh")
        if not isinstance(audio_url, str):
            raise TypeError(f"audio_url must be str, got {type(audio_url).__name__}")
        if not isinstance(language, str):
            raise TypeError(f"language must be str, got {type(language).__name__}")
        try:
            transcript = "[SIM] ASR: 你好，这是一个模拟语音转文字示例。今天天气怎么样？"
            data = {
                "audio_url": audio_url,
                "language": language,
                "segments": [
                    {"start": 0.0, "end": 1.2, "text": "你好，"},
                    {"start": 1.2, "end": 4.5, "text": "这是一个模拟语音转文字示例。"},
                    {"start": 4.5, "end": 6.0, "text": "今天天气怎么样？"},
                ],
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output=transcript,
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("ASRTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )


class TTSTool(BaseTool):
    name: ClassVar[str] = "tool_tts"
    description: ClassVar[str] = "Text-To-Speech: render text to spoken audio."
    modal: ClassVar[str] = "audio"
    inputs_schema: ClassVar[dict] = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize"},
            "voice": {
                "type": "string",
                "description": "Voice preset name (female, male, ...)",
            },
        },
    }
    outputs_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "audio_url": {"type": "string"},
            "duration_ms": {"type": "integer"},
        },
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        text = kwargs.get("text", "")
        voice = kwargs.get("voice", "female")
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not isinstance(voice, str):
            raise TypeError(f"voice must be str, got {type(voice).__name__}")
        try:
            h = _md5(text or "empty")
            duration_ms = max(0, len(text) * 150)
            data = {
                "audio_url": f"https://placeholder.local/sim-tts-{h}.mp3",
                "duration_ms": duration_ms,
                "voice": voice,
                "text": text,
                "simulated": True,
            }
            return ToolResult(
                success=True,
                output="Generated simulated TTS audio",
                data=data,
                tool_name=self.name,
            )
        except Exception as e:
            log.error("TTSTool unexpected error", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
            )
