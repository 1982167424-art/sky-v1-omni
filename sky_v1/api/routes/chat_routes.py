from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request

from sky_v1.api.types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    TextCompletionChoice,
    TextCompletionRequest,
    TextCompletionResponse,
)

router = APIRouter()


def _extract_user_text(messages: list[ChatMessage]) -> str:
    try:
        for m in reversed(messages):
            if m.role == "user":
                if isinstance(m.content, str):
                    return m.content
                if isinstance(m.content, list):
                    parts: list[str] = []
                    for p in m.content:
                        try:
                            if hasattr(p, "text"):
                                parts.append(str(p.text))
                        except Exception:
                            continue
                    if parts:
                        return "\n".join(parts)
    except Exception:
        pass
    return ""


def _sim_answer(prompt: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return "你好，我是 sky-v1-agent，请问有什么可以帮助你的？"
    if "代码" in p or "code" in p.lower() or "print" in p.lower():
        return f"[SIM] 理解了你的问题关于代码：{p[:60]}。我可以帮你运行代码。"
    if "图像" in p or "图片" in p or "image" in p.lower():
        return f"[SIM] 理解了你的图像相关问题：{p[:60]}。我可以帮你生成或理解图片。"
    if "视频" in p or "video" in p.lower():
        return f"[SIM] 理解了你的视频相关问题：{p[:60]}。我可以帮你生成或理解视频。"
    if "3d" in p.lower() or "三维" in p:
        return f"[SIM] 理解了你的3D相关问题：{p[:60]}。我可以帮你生成3D模型。"
    return f"[SIM] sky-v1-agent 响应：{p[:100]}"


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse:
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    prompt_text = _extract_user_text(body.messages)
    finish_reason: str = "stop"
    answer_text: str = ""

    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        try:
            if hasattr(agent, "step"):
                step_result = agent.step(
                    user_message=prompt_text,
                    session_id=body.user or "default",
                )
                if isinstance(step_result, dict):
                    answer_text = str(step_result.get("answer", "")) or _sim_answer(prompt_text)
                else:
                    answer_text = str(step_result) or _sim_answer(prompt_text)
            else:
                answer_text = _sim_answer(prompt_text)
        except Exception as e:
            finish_reason = "error"
            answer_text = f"Agent执行失败：{str(e)[:200]}"
    else:
        finish_reason = "error"
        answer_text = "SkyAgent未配置，请检查依赖安装。"

    usage = ChatCompletionUsage(
        prompt_tokens=len(prompt_text) // 4,
        completion_tokens=len(answer_text) // 4,
        total_tokens=(len(prompt_text) + len(answer_text)) // 4,
    )

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage(role="assistant", content=answer_text),
        finish_reason=finish_reason,  # type: ignore[arg-type]
    )
    return ChatCompletionResponse(
        id=resp_id,
        model=body.model,
        choices=[choice],
        usage=usage,
    )


@router.post("/completions", response_model=TextCompletionResponse)
async def text_completions(
    request: Request,
    body: TextCompletionRequest,
) -> TextCompletionResponse:
    resp_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    prompt_str = body.prompt if isinstance(body.prompt, str) else "\n".join(body.prompt)
    completion_text = f"[SIM Completion]\n{prompt_str[:80]}"
    choice = TextCompletionChoice(
        index=0,
        text=completion_text,
        finish_reason="stop",
    )
    usage = ChatCompletionUsage(
        prompt_tokens=len(prompt_str) // 4,
        completion_tokens=len(completion_text) // 4,
        total_tokens=(len(prompt_str) + len(completion_text)) // 4,
    )
    return TextCompletionResponse(
        id=resp_id,
        model=body.model,
        choices=[choice],
        usage=usage,
    )
