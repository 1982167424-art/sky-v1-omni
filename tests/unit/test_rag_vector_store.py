import pytest


def test_add_then_count(tmp_rag_store, tmp_embedder):
    ids = ["a", "b", "c"]
    texts = ["hello", "world", "foo"]
    embeddings = tmp_embedder.embed_texts(texts)
    tmp_rag_store.add(ids=ids, embeddings=embeddings, documents=texts)
    assert tmp_rag_store.count() == 3


def test_query_shape(tmp_rag_store, tmp_embedder):
    ids = ["d1", "d2", "d3"]
    texts = ["Transformer attention", "NeRF radiance field", "Stable Diffusion"]
    embeddings = tmp_embedder.embed_texts(texts)
    tmp_rag_store.add(ids=ids, embeddings=embeddings, documents=texts)
    q_vec = tmp_embedder.embed_query("attention mechanism")
    results = tmp_rag_store.query(query_embedding=q_vec, top_k=2)
    assert len(results) <= 2
    for r in results:
        assert "id" in r
        assert "distance" in r
        assert "document" in r


def test_add_length_mismatch(tmp_rag_store, tmp_embedder):
    ids = ["a", "b", "c"]
    embeddings = tmp_embedder.embed_texts(["x", "y"])
    with pytest.raises(ValueError):
        tmp_rag_store.add(ids=ids, embeddings=embeddings)


def test_delete_then_zero(tmp_rag_store, tmp_embedder):
    ids = ["a", "b"]
    embeddings = tmp_embedder.embed_texts(["x", "y"])
    tmp_rag_store.add(ids=ids, embeddings=embeddings)
    assert tmp_rag_store.count() == 2
    tmp_rag_store.delete(ids=["a", "b"])
    assert tmp_rag_store.count() == 0
