"""sky_v1.rag.ingestion: 文档数据类、文本分块与批量摄入。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_store import VectorStore
    from .embedding import EmbeddingModel


@dataclass
class Document:
    """文档数据类。"""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    category: str = "general"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Document.id 必须是非空字符串")
        if not isinstance(self.text, str):
            raise ValueError("Document.text 必须是 str")
        if not isinstance(self.metadata, dict):
            raise ValueError("Document.metadata 必须是 dict")
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("Document.category 必须是非空字符串")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """按字符切分文本为重叠块。

    Args:
        text: 原始文本。空字符串返回 []。
        chunk_size: 每块目标字符数。
        chunk_overlap: 相邻块重叠字符数。

    Returns:
        分块后的字符串列表。

    Raises:
        ValueError: chunk_size 或 chunk_overlap 为负值，或 chunk_size <= chunk_overlap。
    """
    if not isinstance(text, str):
        raise ValueError("text 必须是 str")
    if not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("chunk_size 必须是非负整数")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap 必须是非负整数")
    if chunk_size > 0 and chunk_size <= chunk_overlap:
        raise ValueError("chunk_size 必须大于 chunk_overlap")

    if not text:
        return []
    if chunk_size == 0:
        return [text]

    n = len(text)
    if n <= chunk_size:
        return [text]

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = 1

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start += step

    if len(chunks) >= 2:
        tail_len = len(chunks[-1])
        if tail_len < chunk_size // 4:
            merged = chunks[-2] + chunks[-1][chunk_overlap:] if chunk_overlap > 0 else chunks[-2] + chunks[-1]
            chunks = chunks[:-2] + [merged]

    return chunks


def ingest_documents(
    store: "VectorStore",
    embedder: "EmbeddingModel",
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    """批量摄入文档到向量存储。

    对每个文档：chunk_text → 生成稳定 chunk_id = f"{doc.id}#chunk{i}"
    → embed → store.add → 返回统计信息。

    Args:
        store: 目标向量存储。
        embedder: 嵌入模型。
        docs: Document 列表。
        chunk_size: 分块大小。
        chunk_overlap: 分块重叠。

    Returns:
        统计字典: {"total_docs", "total_chunks", "categories": Counter}
    """
    from .vector_store import VectorStore
    from .embedding import EmbeddingModel

    if not isinstance(store, VectorStore):
        raise ValueError("store 必须是 VectorStore 实例")
    if not isinstance(embedder, EmbeddingModel):
        raise ValueError("embedder 必须是 EmbeddingModel 实例")
    if not isinstance(docs, list):
        raise ValueError("docs 必须是 list[Document]")

    total_docs = 0
    total_chunks = 0
    categories: Counter = Counter()

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []
    all_documents: list[str] = []

    for doc in docs:
        if not isinstance(doc, Document):
            raise ValueError("docs 中每个元素必须是 Document 实例")
        total_docs += 1
        categories[doc.category] += 1

        chunks = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            continue

        chunk_ids = [f"{doc.id}#chunk{i}" for i in range(len(chunks))]
        chunk_metadatas: list[dict] = []
        for i in range(len(chunks)):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta.setdefault("category", doc.category)
            meta.setdefault("doc_id", doc.id)
            meta["chunk_index"] = i
            meta["chunk_count"] = len(chunks)
            chunk_metadatas.append(meta)

        embeddings = embedder.embed_texts(chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"embed_texts 返回长度 ({len(embeddings)}) 与 chunks 长度 ({len(chunks)}) 不匹配"
            )

        all_ids.extend(chunk_ids)
        all_embeddings.extend(embeddings)
        all_metadatas.extend(chunk_metadatas)
        all_documents.extend(chunks)
        total_chunks += len(chunks)

    if all_ids:
        store.add(
            ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            documents=all_documents,
        )

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "categories": dict(categories),
    }
