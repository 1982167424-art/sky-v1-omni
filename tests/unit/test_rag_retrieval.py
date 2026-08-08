from sky_v1.rag.retrieval import hyde_rewrite, Reranker, retrieve_top_k
from sky_v1.rag.vector_store import InMemoryStore
from sky_v1.rag.embedding import SimEmbeddingFallback


def test_hyde_prefix():
    r = hyde_rewrite("X")
    assert r.startswith("请给出关于以下问题的详细技术文档和教程：")


def test_rerank_filters_high_distance():
    reranker = Reranker()
    fake_results = [
        {"id": "a", "distance": 0.99, "document": "x"},
        {"id": "b", "distance": 0.96, "document": "y"},
        {"id": "c", "distance": 0.5, "document": "z"},
    ]
    out = reranker.rerank("query", fake_results, threshold=0.95)
    assert len(out) < 3


def test_retrieve_topk_shape():
    store = InMemoryStore()
    emb = SimEmbeddingFallback()
    texts = [
        "Transformer attention mechanism",
        "NeRF neural radiance field",
        "Stable Diffusion image model",
        "Whisper speech recognition",
        "PointNet 3D point cloud",
    ]
    ids = [f"doc{i}" for i in range(len(texts))]
    embs = emb.embed_texts(texts)
    store.add(ids=ids, embeddings=embs, documents=texts)
    results = retrieve_top_k(store, emb, "Transformer", top_k=5)
    assert len(results) <= 5
    for r in results:
        assert "id" in r
        assert "distance" in r
