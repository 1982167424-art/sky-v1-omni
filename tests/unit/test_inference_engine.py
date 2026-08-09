import torch
import pytest
from pathlib import Path
from sky_v1.inference.engine import SkyInferenceEngine
from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig

def _mini_cfg():
    return SkyModelConfig(
        model_name="mini-test",
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
        max_seq_len=128,
        vocab_size=512,
        image_vocab_size=0,
        audio_vocab_size=0,
        video_vocab_size=0,
        three_d_vocab_size=0,
        modal=ModalConfig(),
        heads=HeadsConfig(),
    )

def test_engine_init_and_generate():
    cfg = _mini_cfg()
    engine = SkyInferenceEngine(cfg, device="cpu", dtype="fp32", max_batch_size=2)
    out = engine.generate_text(prompt_ids=torch.tensor([[10, 20, 30]]), max_new_tokens=5, temperature=0.0)
    assert out.token_ids.shape == (1, 5)
    assert out.done is True or out.done is False

def test_engine_chat_compose_and_generate():
    cfg = _mini_cfg()
    engine = SkyInferenceEngine(cfg, device="cpu")
    messages = [{"role": "user", "content": "hello"}]
    out = engine.chat(messages, max_new_tokens=3, temperature=0.0)
    assert "text" in out
    assert isinstance(out["text"], str)

def test_engine_modal_inference_text_image():
    cfg = _mini_cfg()
    engine = SkyInferenceEngine(cfg, device="cpu")
    txt_ids = torch.tensor([[10,20]])
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        preds = engine.predict(text_ids=txt_ids, image=img)
    assert "text_logits" in preds or "text_tokens" in preds
