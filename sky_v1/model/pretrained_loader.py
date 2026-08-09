"""Pretrained warm-start loader — implements spec §2.2 Scheme B.

Each modality tokenizer preferentially loads SOTA pretrained weights as a
warm start. Loading is strictly best-effort: any failure (missing
``transformers`` / ``huggingface_hub``, network error, shape mismatch) is
caught and reported so that training can proceed with random init when no
pretrained weights are available.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Dependency probes (lazy + safe)                                             #
# --------------------------------------------------------------------------- #
def _try_import_transformers():
    try:
        import transformers  # type: ignore
        return transformers
    except Exception:
        return None


def _try_import_huggingface_hub():
    try:
        import huggingface_hub  # type: ignore
        return huggingface_hub
    except Exception:
        return None


def _resolve_hf_path(hf_repo_or_path: str) -> str:
    """Return a local path to the weights, falling back to the repo id.

    If ``hf_repo_or_path`` already points at a local file/dir, return it
    untouched. Otherwise attempt ``huggingface_hub.snapshot_download``; on
    any failure return the original string so that ``from_pretrained`` can
    attempt a direct hub fetch (or fail gracefully upstream).
    """
    if not hf_repo_or_path:
        return hf_repo_or_path
    if os.path.isdir(hf_repo_or_path) or os.path.isfile(hf_repo_or_path):
        return hf_repo_or_path
    hub = _try_import_huggingface_hub()
    if hub is None:
        return hf_repo_or_path
    try:
        local = hub.snapshot_download(repo_id=hf_repo_or_path)
        if isinstance(local, str) and local:
            return local
    except Exception as e:  # pragma: no cover - network/permission errors
        print(f"[pretrained] snapshot_download failed for {hf_repo_or_path}: {e}")
    return hf_repo_or_path


# --------------------------------------------------------------------------- #
# Generic block-copy helper                                                   #
# --------------------------------------------------------------------------- #
def _copy_tensor_block(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Copy the leading overlap of ``src`` into ``dst`` in-place.

    Takes the element-wise minimum extent along every dimension. Returns
    ``True`` if any bytes were copied, ``False`` if shapes are
    incompatible (different rank and unequal numel).
    """
    if dst.dim() != src.dim():
        if dst.numel() == src.numel():
            src = src.reshape(dst.shape)
        else:
            return False
    slices = tuple(slice(0, min(int(d), int(s))) for d, s in zip(dst.shape, src.shape))
    with torch.no_grad():
        dst[slices].copy_(src[slices].to(dst.dtype))
    return True


def _tok_attr(model: Any, name: str) -> Any:
    """Return ``model.<name>`` if present, else ``model`` itself.

    Allows the loaders to accept either a full SkyModel (with ``text_tok`` /
    ``image_tok`` / ``audio_tok`` sub-modules) or a bare tokenizer module.
    """
    obj = getattr(model, name, None)
    return obj if obj is not None else model


# --------------------------------------------------------------------------- #
# Text — Qwen embeddings                                                      #
# --------------------------------------------------------------------------- #
def load_qwen_embeddings_into_text_tokenizer(model: Any, hf_repo_or_path: str) -> bool:
    """Load Qwen ``embed_tokens`` weights into ``TextTokenizer.embedding``.

    Vocab/hidden size mismatches are handled by copying the leading
    ``min(src, dst)`` block. Returns ``True`` on success, ``False``
    otherwise. Never raises.
    """
    try:
        transformers = _try_import_transformers()
        if transformers is None:
            print("[pretrained] WARNING: transformers not installed; skipping Qwen embedding load")
            return False
        local_path = _resolve_hf_path(hf_repo_or_path)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(local_path)

        src_weight: torch.Tensor | None = None
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "embed_tokens"):
            src_weight = hf_model.model.embed_tokens.weight
        elif hasattr(hf_model, "embed_tokens"):
            src_weight = hf_model.embed_tokens.weight
        elif hasattr(hf_model, "get_input_embeddings"):
            emb = hf_model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                src_weight = emb.weight
        if src_weight is None:
            print("[pretrained] WARNING: could not locate embed_tokens in Qwen model")
            return False

        text_tok = _tok_attr(model, "text_tok")
        if not hasattr(text_tok, "embedding"):
            print("[pretrained] WARNING: model has no text_tok.embedding to load into")
            return False
        dst = text_tok.embedding.weight
        ok = _copy_tensor_block(dst, src_weight.detach())
        if not ok:
            print("[pretrained] WARNING: Qwen embedding shape incompatible with text tokenizer")
            return False
        print(
            f"[pretrained] loaded Qwen embeddings into text tokenizer "
            f"({tuple(src_weight.shape)} -> {tuple(dst.shape)})"
        )
        return True
    except Exception as e:
        print(f"[pretrained] WARNING: failed to load Qwen embeddings from {hf_repo_or_path}: {e}")
        return False


# --------------------------------------------------------------------------- #
# Image — CLIP ViT patch_embed                                                #
# --------------------------------------------------------------------------- #
def load_clip_vit_into_image_tokenizer(model: Any, hf_repo_or_path: str) -> bool:
    """Load CLIP ViT ``patch_embed.proj`` (Conv2d) into ``ImageTokenizer.proj``.

    The Conv2d weight ``(hidden, in_ch, ps, ps)`` is reshaped to the
    Linear layout ``(hidden, in_ch*ps*ps)`` before the block copy.
    """
    try:
        transformers = _try_import_transformers()
        if transformers is None:
            print("[pretrained] WARNING: transformers not installed; skipping CLIP ViT load")
            return False
        local_path = _resolve_hf_path(hf_repo_or_path)

        clip_cls = getattr(transformers, "CLIPVisionModel", None)
        if clip_cls is None:
            clip_cls = getattr(transformers, "CLIPModel", None)
        if clip_cls is None:
            print("[pretrained] WARNING: transformers has no CLIPVisionModel/CLIPModel")
            return False
        hf_model = clip_cls.from_pretrained(local_path)

        vision = getattr(hf_model, "vision_model", hf_model)
        patch_embed = getattr(vision, "patch_embed", None)
        if patch_embed is None or not hasattr(patch_embed, "proj"):
            print("[pretrained] WARNING: could not locate vision_model.patch_embed.proj in CLIP")
            return False
        conv = patch_embed.proj  # nn.Conv2d
        conv_w = conv.weight.detach()  # (hidden, in_ch, ps, ps)
        conv_b = getattr(conv, "bias", None)

        image_tok = _tok_attr(model, "image_tok")
        proj = getattr(image_tok, "proj", None)
        if proj is None or not hasattr(proj, "weight"):
            print("[pretrained] WARNING: model has no image_tok.proj to load into")
            return False

        lin_w = conv_w.reshape(conv_w.shape[0], -1)  # (hidden, in_ch*ps*ps)
        ok_w = _copy_tensor_block(proj.weight, lin_w)
        if not ok_w:
            print("[pretrained] WARNING: CLIP patch_embed weight shape incompatible with image tokenizer")
            return False
        if conv_b is not None and hasattr(proj, "bias") and proj.bias is not None:
            _copy_tensor_block(proj.bias, conv_b.detach())
        print(
            f"[pretrained] loaded CLIP ViT patch_embed into image tokenizer "
            f"({tuple(conv_w.shape)} -> {tuple(proj.weight.shape)})"
        )
        return True
    except Exception as e:
        print(f"[pretrained] WARNING: failed to load CLIP ViT from {hf_repo_or_path}: {e}")
        return False


# --------------------------------------------------------------------------- #
# Audio — Whisper encoder conv1/conv2                                         #
# --------------------------------------------------------------------------- #
def load_whisper_encoder_into_audio_tokenizer(model: Any, hf_repo_or_path: str) -> bool:
    """Load Whisper ``encoder.conv1`` (and ``conv2`` if compatible) into
    ``AudioTokenizer.conv``.

    Only the leading overlap of each Conv1d weight/bias is copied so that
    mismatches in ``kernel_size`` / ``out_channels`` do not abort loading.
    """
    try:
        transformers = _try_import_transformers()
        if transformers is None:
            print("[pretrained] WARNING: transformers not installed; skipping Whisper load")
            return False
        local_path = _resolve_hf_path(hf_repo_or_path)
        whisper_cls = getattr(transformers, "WhisperModel", None)
        if whisper_cls is None:
            print("[pretrained] WARNING: transformers has no WhisperModel")
            return False
        hf_model = whisper_cls.from_pretrained(local_path)

        encoder = getattr(hf_model, "encoder", hf_model)
        audio_tok = _tok_attr(model, "audio_tok")
        conv = getattr(audio_tok, "conv", None)
        if conv is None or not hasattr(conv, "weight"):
            print("[pretrained] WARNING: model has no audio_tok.conv to load into")
            return False

        copied_any = False
        for name in ("conv1", "conv2"):
            layer = getattr(encoder, name, None)
            if layer is None or not hasattr(layer, "weight"):
                continue
            w_ok = _copy_tensor_block(conv.weight, layer.weight.detach())
            b = getattr(layer, "bias", None)
            if b is not None and conv.bias is not None:
                _copy_tensor_block(conv.bias, b.detach())
            if w_ok:
                copied_any = True
                print(f"[pretrained] copied Whisper encoder.{name} into audio tokenizer conv")
        if not copied_any:
            print("[pretrained] WARNING: no Whisper encoder conv weights were compatible")
            return False
        return True
    except Exception as e:
        print(f"[pretrained] WARNING: failed to load Whisper encoder from {hf_repo_or_path}: {e}")
        return False


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
_LOADERS = {
    "text": load_qwen_embeddings_into_text_tokenizer,
    "image": load_clip_vit_into_image_tokenizer,
    "audio": load_whisper_encoder_into_audio_tokenizer,
}


def _extract_pretrained_dict(config: Any) -> dict:
    """Normalize a ``pretrained`` mapping out of a config object or dict."""
    pretrained: Any = None
    if hasattr(config, "pretrained"):
        pretrained = config.pretrained
    elif isinstance(config, dict):
        pretrained = config.get("pretrained")
    if isinstance(pretrained, str):
        return {"text": pretrained}
    if isinstance(pretrained, dict):
        return dict(pretrained)
    return {}


def load_all_pretrained(model: nn.Module, config: Any) -> dict[str, str]:
    """Attempt to warm-start every modality listed in ``config.pretrained``.

    Returns a status dict ``{modality: "loaded" | "skipped" | "failed"}``.
    Never raises — every loader is wrapped in its own try/except.
    """
    pretrained = _extract_pretrained_dict(config)
    report: dict[str, str] = {}
    for modal in ("text", "image", "audio"):
        repo = pretrained.get(modal)
        if not repo:
            report[modal] = "skipped"
            continue
        loader = _LOADERS.get(modal)
        if loader is None:
            report[modal] = "skipped"
            continue
        try:
            ok = loader(model, repo)
        except Exception as e:  # defensive — loaders already guard
            print(f"[pretrained] {modal} loader raised unexpectedly: {e}")
            ok = False
        report[modal] = "loaded" if ok else "failed"
    return report
