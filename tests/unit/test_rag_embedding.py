import numpy as np

from sky_v1.rag.embedding import SimEmbeddingFallback


def _cos_sim(a, b):
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va) or 1.0
    nb = np.linalg.norm(vb) or 1.0
    return float(np.dot(va, vb) / (na * nb))


def test_sim_dim():
    assert SimEmbeddingFallback().dim == 64


def test_sim_deterministic():
    emb = SimEmbeddingFallback()
    r1 = emb.embed_texts(["hello"])
    r2 = emb.embed_texts(["hello"])
    assert len(r1) == 1
    assert len(r2) == 1
    sim = _cos_sim(r1[0], r2[0])
    assert sim >= 0.999


def test_sim_empty_list():
    emb = SimEmbeddingFallback()
    result = emb.embed_texts([])
    assert result == []
