from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Request

from sky_v1.api.types import (
    ChatCompletionUsage,
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    RagQueryRequest,
    RagQueryResponse,
)

router = APIRouter()

_EMBED_DIM = 64


def _hash_to_embedding(s: str, dim: int = _EMBED_DIM) -> list[float]:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    vec: list[float] = []
    for i in range(dim):
        b = h[i % len(h)]
        val = (b / 255.0) * 2.0 - 1.0
        vec.append(round(val, 6))
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [round(v / norm, 6) for v in vec]


@router.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(
    request: Request,
    body: RagQueryRequest,
) -> RagQueryResponse:
    results: list[dict[str, Any]] = []
    warning: str | None = None
    rag_kb = getattr(request.app.state, "rag_kb", None)
    if rag_kb is None:
        warning = "KnowledgeBase未配置，请安装 rag 依赖并初始化。"
    else:
        try:
            if hasattr(rag_kb, "query"):
                qr = rag_kb.query(body.query, body.top_k)
                if isinstance(qr, list):
                    for item in qr:
                        if isinstance(item, dict):
                            results.append(item)
                        else:
                            try:
                                results.append({"content": str(item)})
                            except Exception:
                                pass
        except Exception as e:
            warning = f"RAG查询异常：{str(e)[:200]}"

    return RagQueryResponse(
        query=body.query,
        results=results,
        count=len(results),
        warning=warning,
    )


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def embeddings(
    request: Request,
    body: EmbeddingsRequest,
) -> EmbeddingsResponse:
    inputs = body.input if isinstance(body.input, list) else [body.input]
    embedder = None
    try:
        rag_kb = getattr(request.app.state, "rag_kb", None)
        if rag_kb is not None and hasattr(rag_kb, "embedder"):
            embedder = rag_kb.embedder
    except Exception:
        embedder = None

    data: list[EmbeddingObject] = []
    total_tokens = 0
    for idx, text in enumerate(inputs):
        vec: list[float] = []
        if embedder is not None and hasattr(embedder, "embed"):
            try:
                ev = embedder.embed(text)
                if isinstance(ev, (list, tuple)):
                    vec = [float(x) for x in ev]
            except Exception:
                vec = []
        if not vec:
            vec = _hash_to_embedding(text, _EMBED_DIM)
        data.append(EmbeddingObject(object="embedding", index=idx, embedding=vec))
        total_tokens += len(text) // 4

    usage = ChatCompletionUsage(prompt_tokens=total_tokens, total_tokens=total_tokens)
    return EmbeddingsResponse(
        object="list",
        data=data,
        model=body.model,
        usage=usage,
    )


@router.get("/rag/health")
async def rag_health(request: Request) -> dict[str, Any]:
    rag_kb = getattr(request.app.state, "rag_kb", None)
    count = 0
    persist_dir = ""
    try:
        if rag_kb is not None and hasattr(rag_kb, "count"):
            count = int(rag_kb.count())
    except Exception:
        count = 0
    try:
        if rag_kb is not None and hasattr(rag_kb, "persist_directory"):
            persist_dir = str(rag_kb.persist_directory)
    except Exception:
        persist_dir = ""
    return {
        "status": "ok",
        "count": count,
        "persist_dir": persist_dir,
    }
