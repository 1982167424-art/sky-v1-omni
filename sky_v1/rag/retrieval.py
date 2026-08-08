"""sky_v1.rag.retrieval: 检索、HyDE 改写与规则重排。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .vector_store import VectorStore
    from .embedding import EmbeddingModel


def hyde_rewrite(query: str) -> str:
    """极简 HyDE 查询改写（无 LLM 调用，保证离线可用）。

    给 query 增加一个提示前缀，引导检索到更详细的技术文档块。
    """
    if not isinstance(query, str):
        raise ValueError("query 必须是 str")
    prefix = "请给出关于以下问题的详细技术文档和教程："
    if not query.strip():
        return prefix
    return f"{prefix}{query}"


_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = _WORD_RE.findall(text.lower())
    return {t for t in tokens if len(t) > 0}


class Reranker:
    """纯规则重排器，不依赖任何外部模型。

    策略：
    1. 先按 distance 过滤 > threshold 的结果；
    2. 对剩余结果计算 query 与 document 的词 Jaccard 相似度作为加分项；
    3. 最终按 (jaccard_score, -distance) 降序，等价于综合分降序 + 距离升序。
    """

    def rerank(
        self,
        query: str,
        results: list[dict],
        threshold: float = 0.95,
    ) -> list[dict]:
        if not isinstance(query, str):
            raise ValueError("query 必须是 str")
        if not isinstance(results, list):
            raise ValueError("results 必须是 list[dict]")
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold 必须是数值")

        if not results:
            return []

        q_tokens = _tokenize(query)

        kept: list[tuple[float, float, dict]] = []
        for r in results:
            if not isinstance(r, dict):
                raise ValueError("results 每项必须是 dict")
            distance = float(r.get("distance", 1.0))
            if distance > threshold:
                continue
            doc_text = r.get("document", "") or ""
            d_tokens = _tokenize(doc_text)
            if q_tokens and d_tokens:
                inter = len(q_tokens & d_tokens)
                union = len(q_tokens | d_tokens)
                jaccard = inter / union if union > 0 else 0.0
            else:
                jaccard = 0.0
            kept.append((jaccard, distance, r))

        kept.sort(key=lambda x: (-x[0], x[1]))
        return [item[2] for item in kept]


def retrieve_top_k(
    store: "VectorStore",
    embedder: "EmbeddingModel",
    query: str,
    top_k: int = 5,
    reranker: Optional["Reranker"] = None,
    where: Optional[dict] = None,
) -> list[dict]:
    """端到端检索入口。

    流程: hyde_rewrite(query) → embed_query → store.query (取 top_k*2 候选)
    → (可选) reranker.rerank → 截断 top_k。
    """
    from .vector_store import VectorStore
    from .embedding import EmbeddingModel

    if not isinstance(store, VectorStore):
        raise ValueError("store 必须是 VectorStore 实例")
    if not isinstance(embedder, EmbeddingModel):
        raise ValueError("embedder 必须是 EmbeddingModel 实例")
    if not isinstance(query, str):
        raise ValueError("query 必须是 str")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是 >=1 的整数")
    if reranker is not None and not isinstance(reranker, Reranker):
        raise ValueError("reranker 必须是 Reranker 实例或 None")
    if where is not None and not isinstance(where, dict):
        raise ValueError("where 必须是 dict 或 None")

    rewritten = hyde_rewrite(query)
    query_vec = embedder.embed_query(rewritten)

    candidate_k = max(top_k * 2, 10)
    candidates = store.query(query_embedding=query_vec, top_k=candidate_k, where=where)

    if reranker is not None:
        candidates = reranker.rerank(query, candidates)

    return candidates[:top_k]
