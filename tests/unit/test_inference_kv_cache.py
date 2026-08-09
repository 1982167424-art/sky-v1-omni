import torch
import pytest
from sky_v1.inference.kv_cache import PagedKVCache

def test_kv_cache_put_get_shape():
    cache = PagedKVCache(num_layers=2, num_heads=2, head_dim=8, page_size=4, max_pages=16)
    k = torch.randn(2, 2, 3, 8)
    v = torch.randn(2, 2, 3, 8)
    seq_id = 0
    n_written = cache.put(seq_id, k, v, start_pos=0)
    assert n_written == 3
    k_out, v_out, mask = cache.get(seq_id)
    assert k_out.shape == (2, 2, 4, 8)
    assert mask.shape == (2, 4)
    assert mask.sum().item() == 2 * 3

def test_kv_cache_multi_append():
    cache = PagedKVCache(num_layers=2, num_heads=2, head_dim=8, page_size=4, max_pages=32)
    k1 = torch.randn(1, 2, 2, 8)
    v1 = torch.randn(1, 2, 2, 8)
    k2 = torch.randn(1, 2, 2, 8)
    v2 = torch.randn(1, 2, 2, 8)
    cache.put(0, k1, v1, start_pos=0)
    cache.put(0, k2, v2, start_pos=2)
    _, _, mask = cache.get(0)
    assert mask.sum().item() == 4

def test_kv_cache_clear_and_evict():
    cache = PagedKVCache(num_layers=1, num_heads=2, head_dim=4, page_size=2, max_pages=4)
    for i in range(5):
        cache.put(i, torch.randn(1,2,1,4), torch.randn(1,2,1,4))
    assert cache.num_active_seqs() <= 4
