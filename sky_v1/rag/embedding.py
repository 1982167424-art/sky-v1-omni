"""sky_v1.rag.embedding: 文本嵌入模型抽象与实现。"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EmbeddingModel(ABC):
    """嵌入模型抽象基类。"""

    @property
    def dim(self) -> int:
        """返回向量维度。子类必须定义。"""
        raise NotImplementedError("子类必须实现 dim 属性")

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本。

        Args:
            texts: 文本列表。空列表返回空列表，不抛错。

        Returns:
            与 texts 等长的向量列表。
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本。

        默认调用 embed_texts([text])[0]。
        """
        if not isinstance(text, str):
            raise ValueError("text 必须是 str")
        result = self.embed_texts([text])
        if not result:
            return [0.0] * self.dim
        return result[0]


class SimEmbeddingFallback(EmbeddingModel):
    """完全 deterministic 的哈希嵌入回退方案，无需网络/模型。

    实现：对每个 text 取 SHA-256 digest，前 64 byte 每个 byte
    mod 101 / 100.0 - 0.5 得到 64 维向量，L2 归一化后返回。
    """

    @property
    def dim(self) -> int:
        return 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise ValueError("texts 必须是 list[str]")
        if not texts:
            return []
        out: list[list[float]] = []
        for t in texts:
            if not isinstance(t, str):
                raise ValueError("texts 中每个元素必须是 str")
            digest = hashlib.sha256(bytes(t, "utf-8")).digest()
            raw: list[float] = []
            for b in digest[:64]:
                raw.append((b % 101) / 100.0 - 0.5)
            while len(raw) < 64:
                raw.append(0.0)
            vec = np.asarray(raw, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm == 0:
                norm = 1.0
            vec = vec / norm
            out.append([float(x) for x in vec])
        return out


class BGEOnnxEmbedding(EmbeddingModel):
    """基于 sentence-transformers 的 BGE ONNX 嵌入模型（可选）。

    如果缺少 sentence-transformers 依赖，__init__ 会抛出明确的
    RuntimeError: "请 pip install sky-v1-omni[embed]"，但类定义本身不会崩溃。
    """

    @property
    def dim(self) -> int:
        return 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name 必须是非空字符串")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "请 pip install sky-v1-omni[embed]"
            ) from e
        self._model_name = model_name
        self._device = device
        self._model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise ValueError("texts 必须是 list[str]")
        if not texts:
            return []
        for t in texts:
            if not isinstance(t, str):
                raise ValueError("texts 中每个元素必须是 str")
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in vec] for vec in embeddings]
