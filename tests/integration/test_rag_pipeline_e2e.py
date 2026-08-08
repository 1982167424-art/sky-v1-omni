QUERIES = [
    "Transformer",
    "NeRF",
    "Stable Diffusion",
    "Whisper",
    "CogVideo",
    "PointNet",
    "蒸馏",
    "训练",
    "3D",
    "小米",
]


def test_preset_queries_hit_documents(preset_kb):
    for q in QUERIES:
        results = preset_kb.query(q, top_k=5)
        assert isinstance(results, list)
        non_empty_docs = 0
        for r in results:
            doc = r.get("document", "") if isinstance(r, dict) else ""
            if doc and isinstance(doc, str) and len(doc.strip()) > 0:
                non_empty_docs += 1
        assert non_empty_docs >= 2, f"query={q!r} 非空文档数={non_empty_docs} < 2"
