import sys
import types
import torch
import pytest

from sky_v1.model.config import SkyModelConfig, ModalConfig, HeadsConfig, build_model_from_config
from sky_v1.model.modal_tokenizers import TextTokenizer, ImageTokenizer


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _install_fake_transformers(monkeypatch, auto_factory=None, clip_factory=None, whisper_factory=None):
    """Inject a fake ``transformers`` + ``huggingface_hub`` into sys.modules.

    Keeps the tests fully offline: ``snapshot_download`` raises immediately
    and ``from_pretrained`` returns caller-supplied fake model objects.
    """
    fake = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(path, *a, **k):
            if auto_factory is None:
                raise RuntimeError("no auto factory configured")
            return auto_factory()

    fake.AutoModelForCausalLM = _Auto

    class _CLIPVision:
        @staticmethod
        def from_pretrained(path, *a, **k):
            if clip_factory is None:
                raise RuntimeError("no clip factory configured")
            return clip_factory()

    fake.CLIPVisionModel = _CLIPVision

    class _Whisper:
        @staticmethod
        def from_pretrained(path, *a, **k):
            if whisper_factory is None:
                raise RuntimeError("no whisper factory configured")
            return whisper_factory()

    fake.WhisperModel = _Whisper
    monkeypatch.setitem(sys.modules, "transformers", fake)

    hfhub = types.ModuleType("huggingface_hub")

    def _snap(*a, **k):
        raise RuntimeError("offline test")

    hfhub.snapshot_download = _snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", hfhub)


def _mini_sky_model():
    heads = {k: HeadsConfig(vocab_size=128, num_frames=2, num_points=16, point_dim=3,
                            mesh_vertices=8, patch_size=16, mel_bins=8, out_channels=3)
             for k in ["text", "image", "audio", "video", "three_d"]}
    modal = {k: ModalConfig(modal_id=i, image_size=32, frame_size=32, num_frames=2,
                            num_points=16, mesh_vertices=8, patch_size=16, mel_bins=8)
             for i, k in enumerate(["text", "image", "audio", "video", "three_d"])}
    cfg = SkyModelConfig(name="mini", hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=4, intermediate_size=128, vocab_size=128,
                         max_position_embeddings=128, rope_theta=10000.0,
                         rms_norm_eps=1e-6, modal=modal, heads=heads)
    return build_model_from_config(cfg)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_pretrained_loader_imports():
    import importlib
    mod = importlib.import_module("sky_v1.model.pretrained_loader")
    assert hasattr(mod, "load_all_pretrained")
    assert hasattr(mod, "load_qwen_embeddings_into_text_tokenizer")
    assert hasattr(mod, "load_clip_vit_into_image_tokenizer")
    assert hasattr(mod, "load_whisper_encoder_into_audio_tokenizer")


def test_load_qwen_embeddings_shape_align(monkeypatch):
    dst_vocab, dst_hidden = 100, 32
    text_tok = TextTokenizer(vocab_size=dst_vocab, hidden_size=dst_hidden)
    original = text_tok.embedding.weight.detach().clone()

    # Fake Qwen with a larger vocab and matching hidden dim.
    src_vocab, src_hidden = 200, 32
    src_weight = (
        torch.arange(src_vocab * src_hidden, dtype=torch.float32).reshape(src_vocab, src_hidden)
        + 1000.0
    )

    class _Embed:
        def __init__(self, w):
            self.weight = w

    class _Inner:
        def __init__(self, w):
            self.embed_tokens = _Embed(w)

    class _Qwen:
        def __init__(self, w):
            self.model = _Inner(w)

    _install_fake_transformers(monkeypatch, auto_factory=lambda: _Qwen(src_weight))

    from sky_v1.model.pretrained_loader import load_qwen_embeddings_into_text_tokenizer
    ok = load_qwen_embeddings_into_text_tokenizer(text_tok, "fake/qwen-repo")
    assert ok is True

    # The leading min(src,dst) block must equal the source slice (hidden matches).
    assert torch.allclose(
        text_tok.embedding.weight[:dst_vocab, :dst_hidden],
        src_weight[:dst_vocab, :dst_hidden],
    )
    # And must differ from the random init (proves a copy actually happened).
    assert not torch.allclose(
        text_tok.embedding.weight[:dst_vocab, :dst_hidden],
        original[:dst_vocab, :dst_hidden],
    )


def test_load_clip_vit_patch_embed(monkeypatch):
    in_ch, ps, hidden = 3, 4, 16
    cfg = ModalConfig(patch_size=ps, image_size=16, in_channels=in_ch)
    img_tok = ImageTokenizer(cfg, hidden_size=hidden)
    original = img_tok.proj.weight.detach().clone()

    # Fake CLIP Conv2d patch_embed.proj weight (hidden, in_ch, ps, ps) + bias.
    conv_w = (
        torch.arange(hidden * in_ch * ps * ps, dtype=torch.float32)
        .reshape(hidden, in_ch, ps, ps)
        + 100.0
    )
    conv_b = torch.arange(hidden, dtype=torch.float32) + 5.0

    class _Conv:
        def __init__(self, w, b):
            self.weight = w
            self.bias = b

    class _PatchEmbed:
        def __init__(self, conv):
            self.proj = conv

    class _Vision:
        def __init__(self, conv):
            self.patch_embed = _PatchEmbed(conv)

    class _CLIP:
        def __init__(self, vision):
            self.vision_model = vision

    _install_fake_transformers(monkeypatch, clip_factory=lambda: _CLIP(_Vision(_Conv(conv_w, conv_b))))

    from sky_v1.model.pretrained_loader import load_clip_vit_into_image_tokenizer
    ok = load_clip_vit_into_image_tokenizer(img_tok, "fake/clip-vit")
    assert ok is True

    # Conv2d (hidden, in_ch, ps, ps) -> Linear (hidden, in_ch*ps*ps) via reshape.
    expected_lin = conv_w.reshape(hidden, -1)
    assert torch.allclose(img_tok.proj.weight, expected_lin)
    assert torch.allclose(img_tok.proj.bias, conv_b)
    assert not torch.allclose(img_tok.proj.weight, original)


def test_load_all_pretrained_no_crash(monkeypatch):
    import sky_v1.model.pretrained_loader as pl

    # Force transformers to look unavailable so every loader fails gracefully.
    monkeypatch.setattr(pl, "_try_import_transformers", lambda: None)

    model = _mini_sky_model()

    # Bogus repos must not raise — every modality reports "failed".
    config = {"pretrained": {
        "text": "bogus/qwen",
        "image": "bogus/clip",
        "audio": "bogus/whisper",
    }}
    report = pl.load_all_pretrained(model, config)
    assert isinstance(report, dict)
    assert set(report.keys()) == {"text", "image", "audio"}
    assert all(v in ("loaded", "failed", "skipped") for v in report.values())
    assert all(v == "failed" for v in report.values()), report

    # Empty config -> everything skipped, still no crash.
    report2 = pl.load_all_pretrained(model, {})
    assert all(v == "skipped" for v in report2.values()), report2
