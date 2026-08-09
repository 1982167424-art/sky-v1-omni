"""Audio-family tools: ASRTool (speech->text), TTSTool (text->speech).

接入火山引擎「豆包语音合成大模型 2.0」(seed-tts-2.0) 和
「豆包录音文件识别极速版」(volc.bigasr.auc_turbo) 真实在线 API；
未配置 API Key 时自动 fallback 到模拟数据，保证离线/CI 可用。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, ClassVar

from sky_v1.agent.base import BaseTool, ToolContext, ToolResult
from sky_v1.utils.logging import get_logger

log = get_logger("agent.tools.audio")


def _md5(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError(f"_md5 expects str, got {type(s).__name__}")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _get_volc_creds(ctx: ToolContext) -> tuple[str, str, str]:
    """从 ctx.config 或环境变量读取 (app_id, access_token, resource_id)。"""
    cfg = ctx.config or {}
    app_id = cfg.get("VOLC_TTS_APP_ID") or os.getenv("VOLC_TTS_APP_ID", "")
    access_token = cfg.get("VOLC_TTS_ACCESS_TOKEN") or os.getenv("VOLC_TTS_ACCESS_TOKEN", "")
    resource_id = cfg.get("VOLC_TTS_RESOURCE_ID") or os.getenv("VOLC_TTS_RESOURCE_ID", "seed-tts-2.0")
    return app_id, access_token, resource_id


def _get_volc_asr_creds(ctx: ToolContext) -> tuple[str, str]:
    cfg = ctx.config or {}
    app_id = cfg.get("VOLC_ASR_APP_ID") or os.getenv("VOLC_ASR_APP_ID", "")
    access_token = cfg.get("VOLC_ASR_ACCESS_TOKEN") or os.getenv("VOLC_ASR_ACCESS_TOKEN", "")
    return app_id, access_token


def _http_post_json(url: str, headers: dict, body: dict, timeout: float = 30.0) -> dict:
    """轻量 HTTP POST（避免引入 requests 依赖）。"""
    import urllib.request
    import urllib.error

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"_error": f"HTTP {e.code}", "_body": body_text[:500]}
    except Exception as e:
        return {"_error": str(e)}


def _http_get(url: str, headers: dict, timeout: float = 30.0) -> dict:
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"_error": f"HTTP {e.code}", "_body": body_text[:500]}
    except Exception as e:
        return {"_error": str(e)}


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

        app_id, access_token = _get_volc_asr_creds(ctx)
        # 火山录音文件识别极速版：提交 + 轮询
        if app_id and access_token and audio_url.startswith(("http://", "https://")):
            try:
                submit_url = "https://openspeech.bytedance.com/api/v3/auc_turbo/submit"
                submit_headers = {
                    "X-Api-App-Id": app_id,
                    "X-Api-Access-Key": access_token,
                    "X-Api-Resource-Id": "volc.bigasr.auc_turbo",
                    "Content-Type": "application/json",
                }
                submit_body = {
                    "user": {"uid": "sky-v1-omni"},
                    "audio": {"url": audio_url, "format": "wav"},
                    "additions": json.dumps({"language": language}),
                }
                resp = _http_post_json(submit_url, submit_headers, submit_body, timeout=30.0)
                if "_error" in resp:
                    log.warning("ASR submit failed", error=resp["_error"], body=resp.get("_body", ""))
                else:
                    code = resp.get("code", -1)
                    if code == 0:
                        id_str = resp.get("id", "")
                        if id_str:
                            query_url = f"https://openspeech.bytedance.com/api/v3/auc_turbo/query?id={id_str}"
                            query_headers = {
                                "X-Api-App-Id": app_id,
                                "X-Api-Access-Key": access_token,
                                "X-Api-Resource-Id": "volc.bigasr.auc_turbo",
                            }
                            # 轮询：极速版通常 10s 内出结果
                            for _ in range(60):
                                import time
                                time.sleep(1.0)
                                q = _http_get(query_url, query_headers, timeout=15.0)
                                qcode = q.get("code", -1)
                                if qcode == 0 and q.get("text"):
                                    segments = q.get("utterances") or [
                                        {"start": 0, "end": 0, "text": q["text"]}
                                    ]
                                    return ToolResult(
                                        success=True,
                                        output=q["text"],
                                        data={
                                            "audio_url": audio_url,
                                            "language": language,
                                            "segments": segments,
                                            "provider": "volcengine-seed-asr-turbo",
                                        },
                                        tool_name=self.name,
                                    )
                            # 超时，回退到模拟
                            log.warning("ASR poll timeout, fallback to simulated")
                    else:
                        log.warning("ASR submit error", code=code, message=resp.get("message", ""))
            except Exception as e:
                log.error("ASRTool volcengine error", error=str(e))

        # Fallback: 模拟（保证离线可用，与历史行为一致）
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

    # 火山精品音色映射（公共音色名 → speaker id）
    VOICE_MAP: ClassVar[dict] = {
        "female": "zh_female_vv_jupiter_bigtts",
        "vv": "zh_female_vv_jupiter_bigtts",
        "male": "zh_male_yunzhou_jupiter_bigtts",
        "yunzhou": "zh_male_yunzhou_jupiter_bigtts",
        "xiaohe": "zh_female_xiaohe_jupiter_bigtts",
        "xiaotian": "zh_male_xiaotian_jupiter_bigtts",
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if not isinstance(ctx, ToolContext):
            raise TypeError(f"ctx must be ToolContext, got {type(ctx).__name__}")
        text = kwargs.get("text", "")
        voice = kwargs.get("voice", "female")
        response_format = kwargs.get("response_format", "mp3")
        speed = kwargs.get("speed", 1.0)
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not isinstance(voice, str):
            raise TypeError(f"voice must be str, got {type(voice).__name__}")

        app_id, access_token, resource_id = _get_volc_creds(ctx)
        # 火山豆包 TTS 2.0：同步 /api/v3/tts 直接返回 base64 音频
        if app_id and access_token and text:
            try:
                speaker = self.VOICE_MAP.get(voice, self.VOICE_MAP["female"])
                url = "https://openspeech.bytedance.com/api/v3/tts"
                headers = {
                    "X-Api-App-Id": app_id,
                    "X-Api-Access-Key": access_token,
                    "X-Api-Resource-Id": resource_id,
                    "Content-Type": "application/json",
                }
                body = {
                    "user": {"uid": "sky-v1-omni"},
                    "req_params": {
                        "text": text,
                        "speaker": speaker,
                        "audio_params": {
                            "format": response_format,
                            "sample_rate": 24000,
                            "speech_rate": int((float(speed) - 1.0) * 50),
                            "loudness_rate": 0,
                        },
                    },
                }
                resp = _http_post_json(url, headers, body, timeout=60.0)
                if "_error" in resp:
                    log.warning("TTS volcengine error", error=resp["_error"], body=resp.get("_body", ""))
                else:
                    code = resp.get("code", -1)
                    if code == 0:
                        data_block = resp.get("data") or {}
                        audio_b64 = data_block.get("audio", "")
                        duration_ms = int(data_block.get("duration_ms", max(100, len(text) * 80)))
                        if audio_b64:
                            audio_url = f"data:audio/{response_format};base64,{audio_b64}"
                            return ToolResult(
                                success=True,
                                output="Generated TTS audio via Volcengine seed-tts-2.0",
                                data={
                                    "audio_url": audio_url,
                                    "duration_ms": duration_ms,
                                    "voice": voice,
                                    "speaker": speaker,
                                    "text": text,
                                    "provider": "volcengine-seed-tts-2.0",
                                },
                                tool_name=self.name,
                            )
                    else:
                        log.warning("TTS volcengine code error", code=code, message=resp.get("message", ""))
            except Exception as e:
                log.error("TTSTool volcengine error", error=str(e))

        # Fallback: 模拟（保证离线可用）
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
