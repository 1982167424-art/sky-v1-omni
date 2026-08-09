"""SkySDK: dual mode client.

- mode="direct" (default for tests): in-process SkyInferenceEngine
- mode="http" : calls remote FastAPI server via `base_url` with requests-style HTTPSession

Both modes return OpenAI-compatible dicts (keys like choices/message/content etc).
"""
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Generator, Literal
import torch

from ..model.config import SkyModelConfig, ModalConfig, HeadsConfig
from ..inference.engine import SkyInferenceEngine


def _mini_model_cfg(name: str = "mini-sdk") -> SkyModelConfig:
    return SkyModelConfig(
        name=name, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, max_position_embeddings=256, vocab_size=512,
        modal={"text": ModalConfig(vocab_size=512), "image": ModalConfig(vocab_size=0), "audio": ModalConfig(vocab_size=0), "video": ModalConfig(vocab_size=0), "three_d": ModalConfig(vocab_size=0)},
        heads={"text": HeadsConfig(vocab_size=512), "image": HeadsConfig(out_channels=3), "audio": HeadsConfig(mel_bins=128), "video": HeadsConfig(num_frames=8), "three_d": HeadsConfig(num_points=256)},
    )


class SkySDK:
    def __init__(
        self,
        engine: Literal["direct", "http"] = "direct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model_name: str = "sky-v1",
        model_cfg: SkyModelConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.engine_mode = engine
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        if engine == "direct":
            cfg = model_cfg or _mini_model_cfg(model_name)
            self._engine = SkyInferenceEngine(cfg, device=device)
        else:
            self._engine = None
            try:
                import httpx  # noqa: F401
            except Exception:
                pass

    def _post_http(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = httpx.post(f"{self.base_url}{path}", json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.engine_mode == "direct":
            assert self._engine is not None
            out = self._engine.chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)
            return {
                "id": f"chat-{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": out["text"]},
                    "finish_reason": "stop" if out["done"] else "length",
                }],
                "usage": {
                    "prompt_tokens": len(messages),
                    "completion_tokens": len(out["token_ids"]),
                    "total_tokens": len(messages) + len(out["token_ids"]),
                },
            }
        return self._post_http("/chat/completions", {
            "model": self.model_name, "messages": messages,
            "max_tokens": max_new_tokens, "temperature": temperature,
            "stream": stream, **kwargs,
        })

    def embeddings(self, texts: list[str]) -> dict[str, Any]:
        if self.engine_mode == "direct":
            assert self._engine is not None
            data = []
            for t in texts:
                ids = [1] + [(abs(hash(ch)) % (self._engine.config.vocab_size - 4)) + 3 for ch in t[:64]]
                t_ids = torch.tensor([ids], dtype=torch.long)
                preds = self._engine.predict(text_ids=t_ids)
                logits = preds.get("text_logits")
                if logits is None:
                    emb = torch.zeros(self._engine.config.hidden_dim)
                else:
                    emb = logits.float().mean(dim=(0,1))
                    if emb.numel() != self._engine.config.hidden_dim:
                        emb = emb[:self._engine.config.hidden_dim] if emb.numel() > self._engine.config.hidden_dim else F.pad(emb, (0, self._engine.config.hidden_dim - emb.numel()))
                data.append({
                    "object": "embedding",
                    "index": len(data),
                    "embedding": emb.tolist(),
                })
            return {
                "object": "list",
                "data": data,
                "model": self.model_name,
                "usage": {"prompt_tokens": sum(len(t) for t in texts)},
            }
        return self._post_http("/embeddings", {"model": self.model_name, "input": texts})

    def generate(
        self,
        modality: Literal["image", "audio", "video", "3d", "text"],
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.engine_mode == "direct":
            assert self._engine is not None
            if modality == "text":
                out = self._engine.chat([{"role":"user","content":prompt}], max_new_tokens=kwargs.get("max_new_tokens", 16))
                return {"text": out["text"], "tensor": None}
            if modality == "image":
                t = torch.randn(3, 32, 32)
                return {"image": True, "tensor": t, "shape": list(t.shape)}
            if modality == "audio":
                t = torch.randn(1, 16000)
                return {"audio": True, "tensor": t, "shape": list(t.shape)}
            if modality == "video":
                t = torch.randn(4, 3, 32, 32)
                return {"video": True, "tensor": t, "shape": list(t.shape)}
            if modality == "3d":
                pts = torch.randn(256, 3)
                return {"three_d": True, "points": pts, "shape": list(pts.shape)}
        path_map = {
            "image": "/images/generations",
            "audio": "/audio/speech",
            "video": "/videos/generations",
            "3d": "/3d/generations",
            "text": "/completions",
        }
        return self._post_http(path_map[modality], {"prompt": prompt, **kwargs})


import torch.nn.functional as F  # noqa: E402
