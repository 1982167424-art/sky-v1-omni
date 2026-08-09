import torch
import pytest
from sky_v1.model.lora import LoRALinear, mark_lora_targets_, merge_lora_, unload_lora_

def test_lora_linear_shape_and_forward_no_drop():
    base = torch.nn.Linear(16, 32, bias=True)
    lora = LoRALinear(base, rank=4, alpha=8, dropout=0.0)
    x = torch.randn(2, 8, 16)
    y = lora(x)
    assert y.shape == (2, 8, 32)

def test_lora_output_differs_from_base_when_A_random():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 16, bias=False)
    torch.nn.init.eye_(base.weight)
    lora = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    torch.nn.init.normal_(lora.lora_B, mean=0, std=0.1)
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        y_base = base(x)
        y_lora = lora(x)
    assert not torch.allclose(y_base, y_lora, atol=1e-4)

def test_mark_lora_targets_and_merge_unload_cycle():
    from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig
    from sky_v1.model.sky_model import build_model_from_config
    cfg = SkyModelConfig(
        name="lora-mini", hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=128, max_position_embeddings=128, vocab_size=512,
        modal={"text": ModalConfig(vocab_size=512), "image": ModalConfig(vocab_size=0), "audio": ModalConfig(vocab_size=0), "video": ModalConfig(vocab_size=0), "three_d": ModalConfig(vocab_size=0)},
        heads={"text": HeadsConfig(vocab_size=512), "image": HeadsConfig(out_channels=3), "audio": HeadsConfig(mel_bins=128), "video": HeadsConfig(num_frames=8), "three_d": HeadsConfig(num_points=256)},
    )
    model = build_model_from_config(cfg)
    mark_lora_targets_(model, rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "proj"])
    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    assert has_lora
    x = {"text": torch.randint(0, 256, (1, 8))}
    with torch.no_grad():
        y_before = model(x)["text"]
    merge_lora_(model)
    has_any_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    assert not has_any_lora
    with torch.no_grad():
        y_after = model(x)["text"]
    assert torch.allclose(y_before, y_after, atol=2e-4), "merge changed outputs"
