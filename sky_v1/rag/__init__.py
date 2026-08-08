"""sky_v1.rag: Retrieval-Augmented Generation 子系统。

提供向量存储、嵌入模型、文档分块摄入、检索重排以及知识库门面等完整 RAG 能力。
"""

from .vector_store import VectorStore, ChromaStore, InMemoryStore
from .embedding import EmbeddingModel, SimEmbeddingFallback, BGEOnnxEmbedding
from .ingestion import Document, chunk_text, ingest_documents
from .retrieval import hyde_rewrite, Reranker, retrieve_top_k
from .knowledge_base import KnowledgeBase

__all__ = [
    "VectorStore", "ChromaStore", "InMemoryStore",
    "EmbeddingModel", "SimEmbeddingFallback", "BGEOnnxEmbedding",
    "Document", "chunk_text", "ingest_documents",
    "hyde_rewrite", "Reranker", "retrieve_top_k",
    "KnowledgeBase",
]
