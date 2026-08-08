"""Planner: chooses which tool to call based on user message + attachments + history.

Supports a heuristic keyword router (zero network dependency) as the default, with
hooks for plugging in a real LLM planner later.  Never raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sky_v1.agent.base import ToolContext
from sky_v1.utils.logging import get_logger

log = get_logger("agent.planner")


@dataclass
class ToolCallPlan:
    tool_name: str
    tool_kwargs: dict
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tool_kwargs": dict(self.tool_kwargs),
            "reasoning": self.reasoning,
        }


class PlannerLLM:
    """Planner with heuristic router as fallback (and default).

    Config keys:
        mode: "heuristic" | "llm"  (default heuristic)
        llm_endpoint: str  (optional)
        fallback_heuristic: bool  (default True)
    """

    IMAGE_NOUN_KW = ("image", "图片", "图像", "照片", "photo", "img", "picture", "pic")
    VIDEO_NOUN_KW = ("video", "视频", "影片")
    GENERATE_VERB_KW = (
        "生成",
        "画",
        "制作",
        "做",
        "创建",
        "draw",
        "generate",
        "make",
        "create",
        "produce",
    )
    HEURISTIC_IMAGE_UNDERSTAND_KW = (
        "识别图片",
        "识别图像",
        "识别照片",
        "看图",
        "描述图",
        "描述图片",
        "understand image",
        "caption this",
        "描述一下图片",
        "图片识别",
    )
    HEURISTIC_IMAGE_GENERATE_KW = (
        "生成图片",
        "生成图像",
        "生成照片",
        "画一张",
        "画图",
        "generate image",
        "generate a picture",
        "draw picture",
    )
    HEURISTIC_AUDIO_ASR_KW = (
        "语音识别",
        "转文字",
        "识别语音",
        "asr",
        "speech to text",
        "transcribe",
    )
    HEURISTIC_AUDIO_TTS_KW = (
        "tts",
        "语音合成",
        "朗读",
        "读出来",
        "speak",
        "talk",
        "text to speech",
    )
    HEURISTIC_VIDEO_UNDERSTAND_KW = ("视频理解", "视频分析", "video analysis", "描述视频")
    HEURISTIC_VIDEO_GENERATE_KW = (
        "生成视频",
        "做个视频",
        "制作视频",
        "create video",
        "generate video",
        "video generation",
    )
    HEURISTIC_3D_PC_KW = ("point cloud", "点云", "点云生成")
    HEURISTIC_3D_MESH_KW = ("mesh", "网格", "生成mesh", "生成网格", "obj", "glb")
    HEURISTIC_3D_NERF_KW = ("nerf", "神经辐射场", "neural radiance")
    HEURISTIC_CODE_KW = (
        "写代码",
        "代码",
        "运行代码",
        "执行代码",
        "code",
        "run code",
        "python",
        "脚本",
        "编程",
    )
    HEURISTIC_RAG_KW = ("rag", "知识库", "检索", "查资料", "搜索", "search", "retrieve")

    def __init__(self, config: dict | None = None) -> None:
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"config must be dict or None, got {type(config).__name__}")
        mode = config.get("mode", "heuristic")
        if not isinstance(mode, str):
            raise TypeError(f"config['mode'] must be str, got {type(mode).__name__}")
        if mode not in {"heuristic", "llm"}:
            mode = "heuristic"
        self.config: dict = dict(config)
        self.mode: str = mode
        self.llm_endpoint: str | None = self.config.get("llm_endpoint")
        self.fallback_heuristic: bool = bool(self.config.get("fallback_heuristic", True))

    @staticmethod
    def _fallback_chat_plan(user_message: str, reasoning: str = "Fallback chat") -> ToolCallPlan:
        return ToolCallPlan(
            tool_name="tool_chat",
            tool_kwargs={"prompt": user_message},
            reasoning=reasoning,
        )

    def _has_kw(self, text: str, keywords: tuple[str, ...]) -> bool:
        if not isinstance(text, str):
            return False
        lower = text.lower()
        for kw in keywords:
            if not kw:
                continue
            if kw.lower() in lower:
                return True
        return False

    def _heuristic_route(
        self,
        user_message: str,
        attachments: list[dict] | None,
    ) -> ToolCallPlan:
        msg = user_message if isinstance(user_message, str) else ""
        atts = attachments if isinstance(attachments, list) else []

        for att in atts:
            if not isinstance(att, dict):
                continue
            t = att.get("type", "")
            if not isinstance(t, str):
                continue
            tl = t.lower()
            if tl.startswith("image"):
                prompt = msg or att.get("prompt", "请描述这张图片")
                image_url = att.get("url") or att.get("image_url") or ""
                return ToolCallPlan(
                    tool_name="tool_image_understand",
                    tool_kwargs={"image_url": image_url, "prompt": prompt},
                    reasoning=f"Heuristic: attachment type={t} routed to image understanding",
                )
            if tl.startswith("audio"):
                audio_url = att.get("url") or att.get("audio_url") or ""
                language = att.get("language", "zh")
                return ToolCallPlan(
                    tool_name="tool_asr",
                    tool_kwargs={"audio_url": audio_url, "language": language},
                    reasoning=f"Heuristic: attachment type={t} routed to ASR",
                )
            if tl.startswith("video"):
                prompt = msg or att.get("prompt", "请描述这个视频")
                video_url = att.get("url") or att.get("video_url") or ""
                return ToolCallPlan(
                    tool_name="tool_video_understand",
                    tool_kwargs={"video_url": video_url, "prompt": prompt},
                    reasoning=f"Heuristic: attachment type={t} routed to video understanding",
                )

        if self._has_kw(msg, self.HEURISTIC_3D_NERF_KW):
            views = 24
            return ToolCallPlan(
                tool_name="tool_3d_nerf",
                tool_kwargs={"prompt": msg, "views": views},
                reasoning="Heuristic: NeRF keywords -> tool_3d_nerf",
            )
        if self._has_kw(msg, self.HEURISTIC_3D_MESH_KW):
            return ToolCallPlan(
                tool_name="tool_3d_mesh",
                tool_kwargs={"prompt": msg, "format": "obj"},
                reasoning="Heuristic: mesh keywords -> tool_3d_mesh",
            )
        if self._has_kw(msg, self.HEURISTIC_3D_PC_KW):
            return ToolCallPlan(
                tool_name="tool_3d_pointcloud",
                tool_kwargs={"prompt": msg, "n_points": 16384},
                reasoning="Heuristic: point cloud keywords -> tool_3d_pointcloud",
            )

        if self._has_kw(msg, self.HEURISTIC_VIDEO_GENERATE_KW):
            return ToolCallPlan(
                tool_name="tool_video_generate",
                tool_kwargs={"prompt": msg, "duration_s": 5},
                reasoning="Heuristic: video generate keywords -> tool_video_generate",
            )
        if self._has_kw(msg, self.HEURISTIC_VIDEO_UNDERSTAND_KW):
            return ToolCallPlan(
                tool_name="tool_video_understand",
                tool_kwargs={"video_url": "", "prompt": msg},
                reasoning="Heuristic: video understand keywords -> tool_video_understand",
            )

        if self._has_kw(msg, self.HEURISTIC_AUDIO_TTS_KW):
            text = msg
            return ToolCallPlan(
                tool_name="tool_tts",
                tool_kwargs={"text": text, "voice": "female"},
                reasoning="Heuristic: TTS/speak keywords -> tool_tts",
            )
        if self._has_kw(msg, self.HEURISTIC_AUDIO_ASR_KW):
            return ToolCallPlan(
                tool_name="tool_asr",
                tool_kwargs={"audio_url": "", "language": "zh"},
                reasoning="Heuristic: ASR keywords -> tool_asr",
            )

        has_generate_verb = self._has_kw(msg, self.GENERATE_VERB_KW)
        has_image_noun = self._has_kw(msg, self.IMAGE_NOUN_KW)
        has_video_noun = self._has_kw(msg, self.VIDEO_NOUN_KW)

        if has_generate_verb and has_image_noun:
            return ToolCallPlan(
                tool_name="tool_image_generate",
                tool_kwargs={"prompt": msg, "size": "1024x1024", "steps": 20},
                reasoning="Heuristic: generate verb + image noun -> tool_image_generate",
            )
        if has_generate_verb and has_video_noun:
            return ToolCallPlan(
                tool_name="tool_video_generate",
                tool_kwargs={"prompt": msg, "duration_s": 5},
                reasoning="Heuristic: generate verb + video noun -> tool_video_generate",
            )

        if self._has_kw(msg, self.HEURISTIC_IMAGE_GENERATE_KW):
            return ToolCallPlan(
                tool_name="tool_image_generate",
                tool_kwargs={"prompt": msg, "size": "1024x1024", "steps": 20},
                reasoning="Heuristic: image generate keywords -> tool_image_generate",
            )
        if self._has_kw(msg, self.HEURISTIC_IMAGE_UNDERSTAND_KW):
            return ToolCallPlan(
                tool_name="tool_image_understand",
                tool_kwargs={"image_url": "", "prompt": msg},
                reasoning="Heuristic: image understand keywords -> tool_image_understand",
            )

        if has_image_noun:
            return ToolCallPlan(
                tool_name="tool_image_understand",
                tool_kwargs={"image_url": "", "prompt": msg},
                reasoning="Heuristic: image noun -> tool_image_understand (default)",
            )
        if has_video_noun:
            return ToolCallPlan(
                tool_name="tool_video_understand",
                tool_kwargs={"video_url": "", "prompt": msg},
                reasoning="Heuristic: video noun -> tool_video_understand (default)",
            )

        if self._has_kw(msg, self.HEURISTIC_CODE_KW):
            return ToolCallPlan(
                tool_name="tool_code",
                tool_kwargs={"language": "python", "code": msg, "run": True},
                reasoning="Heuristic: code keywords -> tool_code",
            )
        if self._has_kw(msg, self.HEURISTIC_RAG_KW):
            return ToolCallPlan(
                tool_name="tool_rag",
                tool_kwargs={"query": msg, "top_k": 3},
                reasoning="Heuristic: RAG keywords -> tool_rag",
            )

        return ToolCallPlan(
            tool_name="tool_chat",
            tool_kwargs={"prompt": msg},
            reasoning="Heuristic: default to chat",
        )

    def _llm_route(
        self,
        ctx: ToolContext,
        user_message: str,
        attachments: list[dict] | None,
        history: list[dict] | None,
    ) -> ToolCallPlan | None:
        _ = (ctx, user_message, attachments, history)
        return None

    def plan(
        self,
        ctx: ToolContext,
        user_message: str,
        attachments: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> ToolCallPlan:
        if not isinstance(ctx, ToolContext):
            log.warning("plan: ctx not ToolContext, using fallback")
            return self._fallback_chat_plan(
                str(user_message), "Fallback: bad ctx type"
            )
        if not isinstance(user_message, str):
            log.warning("plan: user_message not str, casting")
            user_message = str(user_message)
        try:
            result: ToolCallPlan | None = None
            if self.mode == "llm":
                try:
                    result = self._llm_route(ctx, user_message, attachments, history)
                except Exception as e:
                    log.warning("LLM planner failed, falling back", error=str(e))
                    result = None
            if result is None and (
                self.mode == "heuristic" or self.fallback_heuristic
            ):
                result = self._heuristic_route(user_message, attachments)
            if result is None:
                result = self._fallback_chat_plan(user_message, "Fallback: no plan")
            return result
        except Exception as e:
            log.error("PlannerLLM.plan crashed", error=str(e))
            return self._fallback_chat_plan(user_message, f"Fallback chat: {e}")
