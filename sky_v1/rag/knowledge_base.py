"""sky_v1.rag.knowledge_base: 知识库门面类，整合所有 RAG 子模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .vector_store import VectorStore, ChromaStore, InMemoryStore
from .embedding import EmbeddingModel, SimEmbeddingFallback
from .ingestion import Document, ingest_documents
from .retrieval import Reranker, retrieve_top_k


class KnowledgeBase:
    """知识库门面类：封装存储、嵌入、摄入、检索全流程。"""

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingModel] = None,
        reranker: Optional[Reranker] = None,
        persist_directory: str = "./chroma_data",
    ) -> None:
        if store is not None and not isinstance(store, VectorStore):
            raise ValueError("store 必须是 VectorStore 或 None")
        if embedder is not None and not isinstance(embedder, EmbeddingModel):
            raise ValueError("embedder 必须是 EmbeddingModel 或 None")
        if reranker is not None and not isinstance(reranker, Reranker):
            raise ValueError("reranker 必须是 Reranker 或 None")
        if not isinstance(persist_directory, str) or not persist_directory:
            raise ValueError("persist_directory 必须是非空字符串")

        if store is not None:
            self._store: VectorStore = store
        else:
            try:
                self._store = ChromaStore(persist_directory=persist_directory)
            except Exception:
                # ChromaDB 不可用时 (如 Python 3.14 tokenizers 编译失败)
                # 自动 fallback 到内存存储
                self._store = InMemoryStore()
        self._embedder: EmbeddingModel = embedder if embedder is not None else SimEmbeddingFallback()
        self._reranker: Reranker = reranker if reranker is not None else Reranker()

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def embedder(self) -> EmbeddingModel:
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    def ingest_documents(
        self,
        docs: list[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> dict:
        """摄入文档列表。代理 ingestion.ingest_documents。"""
        if not isinstance(docs, list):
            raise ValueError("docs 必须是 list[Document]")
        return ingest_documents(
            store=self._store,
            embedder=self._embedder,
            docs=docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_presets(self) -> dict:
        """从 presets.PRESET_DOCS 加载所有预置文档并摄入。

        Returns:
            ingest_documents 的统计结果。
        """
        from .presets import PRESET_DOCS

        docs: list[Document] = []
        for category, doc_id, title, path in PRESET_DOCS:
            if not isinstance(path, Path):
                p = Path(path)
            else:
                p = path
            if not p.exists():
                raise FileNotFoundError(f"预置文档不存在: {p}")
            text = p.read_text(encoding="utf-8")
            metadata = {
                "category": category,
                "title": title,
                "preset": True,
                "path": str(p),
            }
            doc = Document(
                id=f"preset_{doc_id}",
                text=text,
                metadata=metadata,
                category=category,
            )
            docs.append(doc)

        return self.ingest_documents(docs)

    def query(self, q: str, top_k: int = 5) -> list[dict]:
        """查询知识库，返回 top_k 结果。"""
        if not isinstance(q, str):
            raise ValueError("q 必须是 str")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须是 >=1 的整数")
        return retrieve_top_k(
            store=self._store,
            embedder=self._embedder,
            query=q,
            top_k=top_k,
            reranker=self._reranker,
        )

    def count(self) -> int:
        """返回当前知识库中的向量总数。"""
        return self._store.count()

    def delete(self, ids: list[str]) -> None:
        """按 ID 列表删除记录。"""
        if not isinstance(ids, list):
            raise ValueError("ids 必须是 list[str]")
        self._store.delete(ids)

    def ingest_file(self, path: Path | str) -> dict:
        """摄入单个文件。"""
        p = Path(path) if not isinstance(path, Path) else path
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        doc = Document(
            id=f"file_{p.stem}_{hash(str(p)) & 0xFFFFFF:x}",
            text=text,
            metadata={"category": "user_file", "title": p.name, "path": str(p)},
            category="user_file",
        )
        return self.ingest_documents([doc])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """query 的别名，兼容 CLI/SDK 调用。"""
        return self.query(query, top_k=k)


# 别名：兼容顶层 __init__ 和 CLI 中使用的 SkyKnowledgeBase 名称
SkyKnowledgeBase = KnowledgeBase
