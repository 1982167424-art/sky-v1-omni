import torch
from sky_v1.model.modal_heads import (
    TextHead, ImageHead, AudioHead, VideoHead, ThreeDHead,
)
from sky_v1.model.config import HeadsConfig

def test_text_head_logits_shape():
    h = TextHead(HeadsConfig(vocab_size=5000), hidden_size=256)
    x = torch.randn(2, 32, 256)
    logits = h(x)
    assert logits.shape == (2, 32, 5000)
    ids = h.sample(logits, top_p=0.9)
    assert ids.shape == (2, 32)
    assert (ids >= 0).all() and (ids < 5000).all()

def test_image_head_shape():
    h = ImageHead(HeadsConfig(out_channels=3, patch_size=16), hidden_size=256, image_size=64)
    x = torch.randn(2, 16, 256)
    im = h(x)
    assert tuple(im.shape) == (2, 3, 64, 64)

def test_audio_head_shape():
    h = AudioHead(HeadsConfig(mel_bins=128), hidden_size=256)
    x = torch.randn(2, 25, 256)
    mel = h(x)
    assert mel.shape[0] == 2 and mel.shape[1] == 128

def test_video_head_shape():
    h = VideoHead(HeadsConfig(num_frames=4, out_channels=3, patch_size=16), hidden_size=256, frame_size=64)
    x = torch.randn(2, 4 * 16, 256)
    v = h(x)
    assert tuple(v.shape) == (2, 4, 3, 64, 64)

def test_threed_head_shape():
    h = ThreeDHead(HeadsConfig(num_points=64, point_dim=3, mesh_vertices=16), hidden_size=256)
    x = torch.randn(2, 256, 256)
    pts, mesh = h(x)
    assert pts.shape == (2, 64, 3)
    assert mesh.shape == (2, 16, 3)
