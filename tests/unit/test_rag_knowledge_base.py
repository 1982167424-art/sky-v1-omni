def test_kb_empty_count_zero(fresh_kb):
    assert fresh_kb.count() == 0


def test_preset_ingest_success(preset_kb):
    assert preset_kb.count() > 0


def test_preset_query_hits_something(preset_kb):
    results = preset_kb.query("Transformer", top_k=3)
    assert isinstance(results, list)
    assert len(results) >= 1
