import torch
import pytest
from sky_v1.model.modal_tokenizers import (
    TextTokenizer, ImageTokenizer, AudioTokenizer, VideoTokenizer, ThreeDTokenizer,
)

def test_text_tokenizer_embeds_shape():
    t = TextTokenizer(vocab_size=5000, hidden_size=256)
    ids = torch.randint(0, 5000, (2, 32))
    out = t(ids)
    assert out.shape == (2, 32, 256)

def test_image_tokenizer_shape():
    from sky_v1.model.config import ModalConfig
    cfg = ModalConfig(patch_size=16, image_size=64, in_channels=3, modal_dim=256)
    im = ImageTokenizer(cfg, hidden_size=256)
    x = torch.randn(2, 3, 64, 64)
    out = im(x)
    assert out.shape == (2, 16, 256)

def test_audio_tokenizer_shape():
    from sky_v1.model.config import ModalConfig
    cfg = ModalConfig(mel_bins=128, subsample=4, modal_dim=256)
    at = AudioTokenizer(cfg, hidden_size=256)
    x = torch.randn(2, 128, 100)
    out = at(x)
    assert out.shape[0] == 2 and out.shape[-1] == 256
    assert out.shape[1] <= 100

def test_video_tokenizer_shape():
    from sky_v1.model.config import ModalConfig
    cfg = ModalConfig(num_frames=4, frame_size=64, patch_size=16, modal_dim=256)
    vt = VideoTokenizer(cfg, hidden_size=256)
    x = torch.randn(2, 4, 3, 64, 64)
    out = vt(x)
    assert out.ndim == 3 and out.shape[0] == 2 and out.shape[-1] == 256

def test_threed_tokenizer_shape():
    from sky_v1.model.config import ModalConfig
    cfg = ModalConfig(num_points=64, point_dim=6, mesh_vertices=16, modal_dim=256)
    t3 = ThreeDTokenizer(cfg, hidden_size=256)
    pts = torch.randn(2, 64, 6)
    verts = torch.randn(2, 16, 3)
    out = t3(pts, verts)
    assert out.ndim == 3 and out.shape[0] == 2 and out.shape[-1] == 256
