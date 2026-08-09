from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from sky_v1 import __version__


_PYDANTIC_CONFIG = {"extra": "ignore", "populate_by_name": True}


class TextContentPart(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["text"]
    text: str


class ImageURLContentPart(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["image_url"]
    image_url: str
    detail: Literal["low", "high", "auto"] = "auto"


class AudioURLContentPart(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["audio_url"]
    audio_url: str


class VideoURLContentPart(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["video_url"]
    video_url: str


class ThreeDURLContentPart(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["3d_url"]
    three_d_url: str


ContentPart = Annotated[
    Union[
        TextContentPart,
        ImageURLContentPart,
        AudioURLContentPart,
        VideoURLContentPart,
        ThreeDURLContentPart,
    ],
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    model_config = _PYDANTIC_CONFIG
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-agent"
    messages: list[ChatMessage]
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=131072)
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = Field(1, ge=1, le=16)
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    model_config = _PYDANTIC_CONFIG
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "error"] = "stop"


class ChatCompletionUsage(BaseModel):
    model_config = _PYDANTIC_CONFIG
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class TextCompletionRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-agent"
    prompt: Union[str, list[str]]
    temperature: float = Field(1.0, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=131072)
    stream: bool = False
    n: int = Field(1, ge=1, le=16)
    user: Optional[str] = None


class TextCompletionChoice(BaseModel):
    model_config = _PYDANTIC_CONFIG
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length", "error"] = "stop"


class TextCompletionResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[TextCompletionChoice]
    usage: ChatCompletionUsage


class EmbeddingsRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-embedding"
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["embedding"] = "embedding"
    index: int = 0
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: ChatCompletionUsage


class RagQueryRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    query: str
    top_k: int = 5
    category: Optional[str] = None


class RagQueryResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    query: str
    results: list[dict[str, Any]]
    count: int
    warning: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-image"
    prompt: str
    n: int = Field(1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"
    user: Optional[str] = None


class ImageObject(BaseModel):
    model_config = _PYDANTIC_CONFIG
    url: str
    revised_prompt: str = ""


class ImageGenerationResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageObject]


class AudioSpeechRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-tts"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)


class AudioSpeechResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    url: str
    duration_ms: int = 0


class AudioTranscriptionRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-asr"
    file_url: str
    language: Optional[str] = None
    response_format: Literal["json", "text", "verbose_json"] = "json"


class AudioTranscriptionResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    text: str
    segments: list[dict[str, Any]] = []


class VideoGenerationRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-video"
    prompt: str
    n: int = Field(1, ge=1, le=5)
    duration_s: int = Field(5, ge=1, le=60)
    size: Optional[str] = None
    user: Optional[str] = None


class VideoObject(BaseModel):
    model_config = _PYDANTIC_CONFIG
    url: str
    duration_s: int = 0


class VideoGenerationResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[VideoObject]


class ThreeDGenerationRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model: str = "sky-v1-3d"
    prompt: str
    n: int = Field(1, ge=1, le=5)
    format_: str = Field("glb", alias="format")
    user: Optional[str] = None


class ThreeDObject(BaseModel):
    model_config = _PYDANTIC_CONFIG
    url: str
    format: str = "glb"


class ThreeDGenerationResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ThreeDObject]


class AgentRunRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    user_message: str
    session_id: str = "default"
    user_id: str = "anonymous"
    attachments: list[dict[str, Any]] = []


class AgentRunResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    answer: str
    tool_calls: list[dict[str, Any]]
    plan: dict[str, Any]
    round: int
    session_id: str


class AgentToolListResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    tools: list[dict[str, Any]]


class HealthResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    status: str = "ok"
    version: str = __version__
    uptime_s: float = 0.0
    rag_count: int = 0
    agent_ok: bool = False


class MetricsResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    requests_total: int = 0
    errors_total: int = 0
    avg_latency_ms: float = 0.0
    components: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 搜索 / 深度推理
# ---------------------------------------------------------------------------
_SEARCH_PROVIDER_LITERAL = Literal["google", "baidu", "bing", "toutiao", "tavily"]


class WebSearchProviderStatus(BaseModel):
    model_config = _PYDANTIC_CONFIG
    provider: str
    available: bool
    reason: str = ""
    count: int = 0
    latency_ms: int = 0


class WebSearchRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    query: str
    num_results: int = Field(default=5, ge=1, le=20)
    providers: list[_SEARCH_PROVIDER_LITERAL] = Field(
        default_factory=lambda: ["google", "baidu", "bing", "toutiao"],
        description="指定 Provider 白名单（Google/百度/Bing/今日头条 并发默认全开）",
    )
    freshness: Literal["any", "day", "week", "month"] = "any"
    skip_cache: bool = False
    allow_simulated: bool = Field(
        default=False,
        description="开发模式：真实搜索全部失败时是否允许返回模拟结果（默认 False）",
    )


class WebSearchResult(BaseModel):
    model_config = _PYDANTIC_CONFIG
    title: str = ""
    url: str = ""
    snippet: str = ""
    provider: str = ""


class WebSearchResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    results: list[WebSearchResult]
    provider: str
    providers_used: list[str]
    provider_statuses: list[WebSearchProviderStatus]
    status: Literal["OK", "PARTIAL", "EMPTY", "FAILED"]
    cached: bool
    simulated: bool
    latency_ms: int


class DeepReasoningRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    question: str
    max_iterations: int = Field(default=3, ge=1, le=6)
    enable_web_search: bool = True
    citations_needed: bool = True


class DeepReasoningResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    plan: list[str]
    iterations: list[dict[str, Any]]
    final_answer: str
    confidence: float
    simulated: bool
    latency_ms: int
