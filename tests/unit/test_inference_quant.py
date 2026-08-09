import torch
import pytest
from sky_v1.inference.quant import (
    W8A8Linear, W4A16Linear, quantize_model_, dequantize_model_
)
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig
from sky_v1.model.sky_model import build_model_from_config

def _cfg():
    return SkyModelConfig(
        model_name="q-mini", hidden_dim=32, num_layers=1, num_heads=2,
        ffn_dim=64, max_seq_len=64, vocab_size=256, image_vocab_size=0,
        audio_vocab_size=0, video_vocab_size=0, three_d_vocab_size=0,
        modal=ModalConfig(), heads=HeadsConfig(),
    )

def test_w8a8_linear_forward_matches():
    fc = torch.nn.Linear(16, 16, bias=True).eval()
    x = torch.randn(2, 4, 16)
    y_ref = fc(x)
    qfc = W8A8Linear.from_float(fc, group_size=8)
    y_q = qfc(x)
    assert y_q.shape == y_ref.shape
    rel = (y_q - y_ref).abs().mean() / max(y_ref.abs().mean(), 1e-6)
    assert rel < 0.6, f"rel_err too large: {rel}"

def test_quantize_model_w8a8_runs():
    cfg = _cfg()
    model = build_model_from_config(cfg).eval()
    quantize_model_(model, {"mode": "w8a8", "group_size": 8})
    has_w8 = any(isinstance(m, W8A8Linear) for m in model.modules())
    assert has_w8
    x = {"text": torch.randint(0, 256, (1, 8))}
    with torch.no_grad():
        out = model(x)
    assert "text" in out or "text_logits" in out

def test_dequantize_roundtrip():
    fc = torch.nn.Linear(8, 8, bias=True).eval()
    x = torch.randn(1, 2, 8)
    y_ref = fc(x)
    qfc = W8A8Linear.from_float(fc, group_size=4)
    dq = qfc.to_float()
    y_dq = dq(x)
    rel = (y_dq - y_ref).abs().mean() / max(y_ref.abs().mean(), 1e-6)
    assert rel < 0.6

def test_quant_mode_none_noop():
    cfg = _cfg()
    model = build_model_from_config(cfg).eval()
    before = {n: p.shape for n, p in model.named_parameters()}
    quantize_model_(model, {"mode": "none"})
    after = {n: p.shape for n, p in model.named_parameters()}
    assert before == after
