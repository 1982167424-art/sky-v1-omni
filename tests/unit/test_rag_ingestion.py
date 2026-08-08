from collections import Counter

import pytest

from sky_v1.rag.ingestion import chunk_text, ingest_documents, Document
from sky_v1.rag.vector_store import InMemoryStore
from sky_v1.rag.embedding import SimEmbeddingFallback


def test_chunk_sizes():
    text = "a" * 50
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert len(chunks) >= 1
    tail_len = len(chunks[-1])
    assert tail_len >= (10 // 4)


def test_chunk_empty_string():
    assert chunk_text("") == []


def test_chunk_negative():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=-1, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, chunk_overlap=-1)


def test_ingest_stats():
    store = InMemoryStore()
    emb = SimEmbeddingFallback()
    docs = [
        Document(id="d1", text="hello world " * 30, category="cat_a"),
        Document(id="d2", text="foo bar baz " * 30, category="cat_b"),
        Document(id="d3", text="spam eggs " * 30, category="cat_a"),
    ]
    stats = ingest_documents(store, emb, docs)
    assert stats["total_docs"] == 3
    assert stats["total_chunks"] >= 3
    cats = stats["categories"]
    assert cats.get("cat_a", 0) == 2
    assert cats.get("cat_b", 0) == 1
