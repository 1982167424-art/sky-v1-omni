import pytest
from sky_v1.sdk.client import SkySDK

def test_sdk_chat_openai_compatible_shape(tmp_path):
    """SDK supports a direct-engine mode (no HTTP) using SkyInferenceEngine for tests."""
    sdk = SkySDK(engine="direct", model_name="mini-sdk")
    resp = sdk.chat_completions(messages=[{"role": "user", "content": "hi"}], max_new_tokens=3)
    assert "choices" in resp
    assert len(resp["choices"]) == 1
    msg = resp["choices"][0]["message"]
    assert "role" in msg and "content" in msg

def test_sdk_embeddings_output_shape():
    sdk = SkySDK(engine="direct", model_name="mini-sdk")
    out = sdk.embeddings(["hello world", "second sentence"])
    assert isinstance(out["data"], list)
    assert len(out["data"]) == 2
    assert isinstance(out["data"][0]["embedding"], list)
    assert len(out["data"][0]["embedding"]) > 0

def test_sdk_generate_modal_text_to_image():
    sdk = SkySDK(engine="direct", model_name="mini-sdk")
    resp = sdk.generate(modality="image", prompt="cat")
    assert "image" in resp or "url" in resp or "tensor" in resp
