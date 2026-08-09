import torch
import pytest
from sky_v1.model.attention import MultiHeadAttention, scaled_dot_product_attention_safe

def test_scaled_dot_product_attention_safe_shape_and_nonnan():
    b, h, sq, sk, d = 2, 4, 16, 20, 32
    q = torch.randn(b, h, sq, d)
    k = torch.randn(b, h, sk, d)
    v = torch.randn(b, h, sk, d)
    out = scaled_dot_product_attention_safe(q, k, v)
    assert out.shape == (b, h, sq, d)
    assert not torch.isnan(out).any()
    mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=1)
    out2 = scaled_dot_product_attention_safe(q, k, v, attn_mask=mask)
    assert out2.shape == out.shape
    assert not torch.isnan(out2).any()

def test_multi_head_attention_forward_backward():
    b, s, h = 2, 32, 8
    hidden = 512
    m = MultiHeadAttention(hidden_size=hidden, num_heads=h, dropout_p=0.0)
    x = torch.randn(b, s, hidden, requires_grad=True)
    out = m(x, x, x)
    assert out.shape == (b, s, hidden)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
