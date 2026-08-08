from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from sky_v1.api.types import (
    AgentRunRequest,
    AgentRunResponse,
    AgentToolListResponse,
)

router = APIRouter()

_SIM_TOOL_REGISTRY = [
    {"name": "tool_chat", "enabled": True, "type": "llm", "description": "通用聊天/推理LLM"},
    {"name": "tool_code", "enabled": True, "type": "code", "description": "Python代码沙箱执行"},
    {"name": "tool_rag", "enabled": True, "type": "rag", "description": "知识库检索问答"},
    {"name": "tool_image_understand", "enabled": True, "type": "vision", "description": "图像理解/描述/OCR"},
    {"name": "tool_image_generate", "enabled": True, "type": "image_gen", "description": "文本生成图像"},
    {"name": "tool_asr", "enabled": True, "type": "audio", "description": "语音转文字（ASR）"},
    {"name": "tool_tts", "enabled": True, "type": "audio", "description": "文字转语音（TTS）"},
    {"name": "tool_video_understand", "enabled": True, "type": "video", "description": "视频内容理解"},
    {"name": "tool_video_generate", "enabled": True, "type": "video_gen", "description": "文本生成视频"},
    {"name": "tool_3d_pointcloud", "enabled": True, "type": "3d", "description": "3D点云处理"},
    {"name": "tool_3d_mesh", "enabled": True, "type": "3d", "description": "3D网格模型生成/转换"},
    {"name": "tool_3d_nerf", "enabled": True, "type": "3d", "description": "NeRF神经辐射场重建"},
]


@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(
    request: Request,
    body: AgentRunRequest,
) -> AgentRunResponse:
    answer = ""
    tool_calls: list[dict[str, Any]] = []
    plan: dict[str, Any] = {}
    round_ = 0

    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        try:
            if hasattr(agent, "step"):
                step_result = agent.step(
                    user_message=body.user_message,
                    session_id=body.session_id,
                    user_id=body.user_id,
                    attachments=body.attachments,
                )
                if isinstance(step_result, dict):
                    answer = str(step_result.get("answer", ""))
                    tc = step_result.get("tool_calls")
                    if isinstance(tc, list):
                        for t in tc:
                            if isinstance(t, dict):
                                tool_calls.append(t)
                            else:
                                try:
                                    tool_calls.append({"tool": str(t)})
                                except Exception:
                                    pass
                    p = step_result.get("plan")
                    if isinstance(p, dict):
                        plan = p
                    else:
                        plan = {"raw": str(p) if p else ""}
                    round_ = int(step_result.get("round", 1))
                else:
                    answer = str(step_result)
                    round_ = 1
        except Exception as e:
            answer = f"Agent执行错误：{str(e)[:300]}"
            round_ = 0

    if not answer:
        answer = f"[SIM Agent] 收到：{body.user_message[:120]}\nAgent未配置，使用模拟响应。"
        plan = {"steps": ["parse_user_intent", "simulate_answer"], "strategy": "simulated"}
        round_ = 1
        tool_calls = [
            {"tool": "tool_chat", "status": "simulated", "input": body.user_message[:50], "output": answer[:80]}
        ]

    return AgentRunResponse(
        answer=answer,
        tool_calls=tool_calls,
        plan=plan,
        round=round_,
        session_id=body.session_id,
    )


@router.get("/agent/tools", response_model=AgentToolListResponse)
async def agent_tools(request: Request) -> AgentToolListResponse:
    tools: list[dict[str, Any]] = []
    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        try:
            if hasattr(agent, "list_tools"):
                lt = agent.list_tools()
                if isinstance(lt, list):
                    for t in lt:
                        if isinstance(t, dict):
                            tools.append(t)
                        else:
                            try:
                                tools.append({"name": str(t)})
                            except Exception:
                                pass
        except Exception:
            tools = []
    if not tools:
        tools = [dict(t) for t in _SIM_TOOL_REGISTRY]
    return AgentToolListResponse(tools=tools)
