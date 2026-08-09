from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any

from ..model.config import SkyModelConfig
from ..model.sky_model import SkyModel, build_model_from_config
from .kv_cache import PagedKVCache


def text_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=False)
    logits = logits / float(temperature)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        min_vals = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < min_vals, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, mask)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    original_shape = probs.shape[:-1]
    sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
    return sampled.view(*original_shape)


@dataclass
class GenerateResult:
    token_ids: torch.Tensor
    logprobs: torch.Tensor | None = None
    done: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


class SkyInferenceEngine:
    def __init__(
        self,
        config: SkyModelConfig | str | Path,
        device: str = "cpu",
        dtype: str = "fp32",
        checkpoint_path: str | Path | None = None,
        max_batch_size: int = 4,
        kv_cache_pages: int = 1024,
        kv_cache_page_size: int = 32,
        quant_config: dict[str, Any] | None = None,
        lora_config: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            from ..model.config import load_config_from_yaml
            config = load_config_from_yaml(config)
        self.config: SkyModelConfig = config
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        _dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.dtype: torch.dtype = _dtype_map.get(dtype, torch.float32)
        if self.dtype == torch.float16 and self.device.type == "cpu":
            self.dtype = torch.float32
        self.max_batch_size = max_batch_size
        self.quant_config = quant_config or {"mode": "none"}
        self.lora_config = lora_config or {"enabled": False}

        self.model: SkyModel = build_model_from_config(self.config).to(self.device).to(self.dtype)
        self.model.eval()
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            self.model.load_state_dict(ckpt, strict=False)

        if self.quant_config.get("mode") not in (None, "none"):
            try:
                from .quant import quantize_model_
                quantize_model_(self.model, self.quant_config)
            except Exception as _exc:
                self.quant_config = {"mode": "none", "_fallback_reason": str(_exc)}

        self.kv_cache = PagedKVCache(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.hidden_dim // self.config.num_heads,
            page_size=kv_cache_page_size,
            max_pages=kv_cache_pages,
            dtype=self.dtype,
            device=str(self.device),
        )
        self._next_seq_id = 0

    def _alloc_seq(self) -> int:
        sid = self._next_seq_id
        self._next_seq_id += 1
        return sid

    @torch.no_grad()
    def _forward_backbone_text_only(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        three_d: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        inputs: dict[str, Any] = {}
        inputs["text"] = text_ids.to(self.device)
        if images is not None:
            inputs["image"] = images.to(self.device, self.dtype)
        if audio is not None:
            inputs["audio"] = audio.to(self.device, self.dtype)
        if video is not None:
            inputs["video"] = video.to(self.device, self.dtype)
        if three_d is not None:
            inputs["three_d"] = {k: v.to(self.device, self.dtype) for k, v in three_d.items()}
        outputs = self.model(inputs)
        return outputs.get("text_logits") or outputs.get("text")

    def generate_text(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        stop_token_ids: list[int] | None = None,
    ) -> GenerateResult:
        B, T = prompt_ids.shape
        if stop_token_ids is None:
            stop_token_ids = [self.config.eos_token_id or 0]
        generated = []
        cur_ids = prompt_ids.to(self.device)
        done_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self._forward_backbone_text_only(cur_ids)[:, -1:, :]
                next_tok = text_sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                generated.append(next_tok.cpu())
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                for i in range(B):
                    if next_tok[i, 0].item() in stop_token_ids:
                        done_mask[i] = True
                if done_mask.all():
                    break
        ids = torch.cat(generated, dim=1)
        return GenerateResult(token_ids=ids, done=bool(done_mask.all()))

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if "stop_token_ids" not in kwargs and self.config.eos_token_id is None:
            kwargs["stop_token_ids"] = [0]
        text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        vocab = self.config.vocab_size
        ids = [1] + [(abs(hash(ch)) % (vocab - 2)) + 3 for ch in text[:128]]
        prompt = torch.tensor([ids], dtype=torch.long, device=self.device)
        gen = self.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
        inv = {i: ch for i, ch in zip(ids, text)}
        out_chars = []
        for tok in gen.token_ids[0].tolist():
            out_chars.append(inv.get(tok, "") if tok in inv else (chr(tok % 0x10FFFF) if tok < 0x110000 else ""))
        out_text = "".join(out_chars).strip() or "(stub)"
        return {"text": out_text, "token_ids": gen.token_ids[0].tolist(), "done": gen.done}

    def predict(
        self,
        text_ids: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        three_d: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        T = 0
        if text_ids is not None:
            inputs["text"] = text_ids.to(self.device)
            T = text_ids.shape[1]
        if image is not None:
            inputs["image"] = image.to(self.device, self.dtype)
        if audio is not None:
            inputs["audio"] = audio.to(self.device, self.dtype)
        if video is not None:
            inputs["video"] = video.to(self.device, self.dtype)
        if three_d is not None:
            inputs["three_d"] = {k: v.to(self.device, self.dtype) for k, v in three_d.items()}
        with torch.no_grad():
            out = self.model(inputs)
        text_logits = out.get("text_logits") or out.get("text")
        return {
            "text_logits": text_logits,
            "text_tokens": text_logits.argmax(-1) if text_logits is not None else None,
            "image_recon": out.get("image_logits") or out.get("image"),
            "audio_recon": out.get("audio_logits") or out.get("audio"),
            "video_recon": out.get("video_logits") or out.get("video"),
            "three_d_recon": out.get("three_d_logits") or out.get("three_d"),
        }
