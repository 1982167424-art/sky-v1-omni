"""sky_v1.rag.vector_store: 向量存储抽象与实现。

提供 VectorStore ABC，以及 ChromaStore (持久化) 与 InMemoryStore (内存/测试) 两种实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import numpy as np


class VectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        documents: Optional[list[str]] = None,
    ) -> None:
        """向存储中添加向量及关联元数据。

        Args:
            ids: 每条记录的唯一 ID 列表。
            embeddings: 与 ids 一一对应的向量列表。
            metadatas: 可选，与 ids 等长的元数据列表。
            documents: 可选，与 ids 等长的原始文本列表。

        Raises:
            ValueError: ids / embeddings / metadatas / documents 长度不一致时。
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """根据查询向量检索最相似的 top_k 条记录。

        Args:
            query_embedding: 查询向量。
            top_k: 返回结果数量。
            where: 可选的元数据过滤条件。

        Returns:
            列表，每项为 {"id", "distance", "metadata", "document"} 字典，
            按 distance 升序排列。
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 ID 删除记录。

        Args:
            ids: 要删除的 ID 列表。
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """返回存储中的记录总数。"""
        ...

    @staticmethod
    def _validate_add_args(
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]],
        documents: Optional[list[str]],
    ) -> None:
        """Guard Clause: 校验 add 参数长度对齐。"""
        if not isinstance(ids, list):
            raise ValueError("ids 必须是 list[str]")
        if not isinstance(embeddings, list):
            raise ValueError("embeddings 必须是 list[list[float]]")
        n = len(ids)
        if len(embeddings) != n:
            raise ValueError(
                f"ids 长度 ({n}) 与 embeddings 长度 ({len(embeddings)}) 不一致"
            )
        if metadatas is not None:
            if not isinstance(metadatas, list):
                raise ValueError("metadatas 必须是 list[dict] 或 None")
            if len(metadatas) != n:
                raise ValueError(
                    f"ids 长度 ({n}) 与 metadatas 长度 ({len(metadatas)}) 不一致"
                )
        if documents is not None:
            if not isinstance(documents, list):
                raise ValueError("documents 必须是 list[str] 或 None")
            if len(documents) != n:
                raise ValueError(
                    f"ids 长度 ({n}) 与 documents 长度 ({len(documents)}) 不一致"
                )


class ChromaStore(VectorStore):
    """基于 ChromaDB 持久化向量存储。"""

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "sky_v1_knowledge",
    ) -> None:
        if not isinstance(persist_directory, str) or not persist_directory:
            raise ValueError("persist_directory 必须是非空字符串")
        if not isinstance(collection_name, str) or not collection_name:
            raise ValueError("collection_name 必须是非空字符串")

        import chromadb  # noqa: F401  (delay import)
        from chromadb import PersistentClient

        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client = PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        documents: Optional[list[str]] = None,
    ) -> None:
        self._validate_add_args(ids, embeddings, metadatas, documents)
        if not ids:
            return
        kwargs: dict = {
            "ids": ids,
            "embeddings": embeddings,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if documents is not None:
            kwargs["documents"] = documents
        self._collection.add(**kwargs)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        if not isinstance(query_embedding, Sequence) or not query_embedding:
            raise ValueError("query_embedding 必须是非空序列")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须是 >=1 的整数")

        total = self.count()
        if total == 0:
            return []
        effective_k = min(top_k, total)

        kwargs: dict = {
            "query_embeddings": [list(query_embedding)],
            "n_results": effective_k,
        }
        if where is not None:
            if not isinstance(where, dict):
                raise ValueError("where 必须是 dict 或 None")
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        ids_list = result.get("ids", [[]])[0] if result.get("ids") else []
        dist_list = result.get("distances", [[]])[0] if result.get("distances") else []
        meta_list = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
        doc_list = result.get("documents", [[]])[0] if result.get("documents") else []

        out: list[dict] = []
        for i in range(len(ids_list)):
            out.append({
                "id": ids_list[i],
                "distance": float(dist_list[i]) if i < len(dist_list) else 0.0,
                "metadata": meta_list[i] if i < len(meta_list) else {},
                "document": doc_list[i] if i < len(doc_list) else "",
            })
        return out

    def delete(self, ids: list[str]) -> None:
        if not isinstance(ids, list):
            raise ValueError("ids 必须是 list[str]")
        if not ids:
            return
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return int(self._collection.count())


class InMemoryStore(VectorStore):
    """纯内存实现，基于 numpy 余弦相似度。用于单元测试和离线环境。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadatas: list[dict] = []
        self._documents: list[str] = []
        self._index: dict[str, int] = {}

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        documents: Optional[list[str]] = None,
    ) -> None:
        self._validate_add_args(ids, embeddings, metadatas, documents)
        if not ids:
            return

        meta_to_use = metadatas if metadatas is not None else [{} for _ in ids]
        docs_to_use = documents if documents is not None else ["" for _ in ids]

        for i, (vid, vec, meta, doc) in enumerate(zip(ids, embeddings, meta_to_use, docs_to_use)):
            if vid in self._index:
                pos = self._index[vid]
                self._embeddings[pos] = np.asarray(vec, dtype=np.float64)
                self._metadatas[pos] = meta
                self._documents[pos] = doc
            else:
                self._index[vid] = len(self._ids)
                self._ids.append(vid)
                self._embeddings.append(np.asarray(vec, dtype=np.float64))
                self._metadatas.append(meta)
                self._documents.append(doc)

    def _matches_where(self, metadata: dict, where: Optional[dict]) -> bool:
        if where is None:
            return True
        for k, v in where.items():
            if metadata.get(k) != v:
                return False
        return True

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        if not isinstance(query_embedding, Sequence) or not query_embedding:
            raise ValueError("query_embedding 必须是非空序列")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须是 >=1 的整数")

        if not self._embeddings:
            return []

        q = np.asarray(query_embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0

        candidates: list[tuple[float, int]] = []
        for i, emb in enumerate(self._embeddings):
            if not self._matches_where(self._metadatas[i], where):
                continue
            e_norm = np.linalg.norm(emb)
            if e_norm == 0:
                e_norm = 1.0
            cos_sim = float(np.dot(q, emb) / (q_norm * e_norm))
            distance = 1.0 - cos_sim
            candidates.append((distance, i))

        candidates.sort(key=lambda x: x[0])
        top = candidates[:top_k]

        return [
            {
                "id": self._ids[idx],
                "distance": dist,
                "metadata": self._metadatas[idx],
                "document": self._documents[idx],
            }
            for dist, idx in top
        ]

    def delete(self, ids: list[str]) -> None:
        if not isinstance(ids, list):
            raise ValueError("ids 必须是 list[str]")
        if not ids:
            return
        to_remove_positions: list[int] = []
        for vid in ids:
            if vid in self._index:
                to_remove_positions.append(self._index[vid])
        if not to_remove_positions:
            return
        to_remove_positions = sorted(set(to_remove_positions), reverse=True)
        for pos in to_remove_positions:
            del self._ids[pos]
            del self._embeddings[pos]
            del self._metadatas[pos]
            del self._documents[pos]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {vid: i for i, vid in enumerate(self._ids)}

    def count(self) -> int:
        return len(self._ids)
