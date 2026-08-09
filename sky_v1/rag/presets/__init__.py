"""sky_v1.rag.presets: 预置知识库条目列表。"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent


PRESET_DOCS: list[tuple[str, str, str, Path]] = [
    (
        "model_architecture",
        "model_architecture_knowledge",
        "Transformer 注意力机制与 FlashAttention、LoRA 原理",
        _HERE / "model_architecture_knowledge.md",
    ),
    (
        "training",
        "training_knowledge",
        "大模型三阶段训练流程简介",
        _HERE / "training_knowledge.md",
    ),
    (
        "distillation",
        "distillation_knowledge",
        "五老师知识蒸馏（KL+CE+Feature Loss）技术详解",
        _HERE / "distillation_knowledge.md",
    ),
    (
        "image_modal",
        "image_modal_knowledge",
        "图像模态：Stable Diffusion、FLUX、DiT 简介",
        _HERE / "image_modal_knowledge.md",
    ),
    (
        "audio_modal",
        "audio_modal_knowledge",
        "音频模态：Whisper ASR、TTS 与 HiFi-GAN 简介",
        _HERE / "audio_modal_knowledge.md",
    ),
    (
        "video_modal",
        "video_modal_knowledge",
        "视频模态：Sora、CogVideo、VideoLLaMA 简介",
        _HERE / "video_modal_knowledge.md",
    ),
    (
        "three_d",
        "three_d_knowledge",
        "3D 模态：PointNet、NeRF、3DGS 与 Mesh 简介",
        _HERE / "three_d_knowledge.md",
    ),
    (
        "github_repos",
        "github_repos_overview",
        "sky-v1-omni 生态 13 个 GitHub 仓库索引简介",
        _HERE / "github_repos_overview.md",
    ),
    (
        "chinese_llm_landscape",
        "chinese_llm_landscape",
        "2026 国内外主流大模型差异化优势（豆包/文心/Qwen/混元/盘古/星火/DeepSeek/Kimi/…）",
        _HERE / "chinese_llm_landscape.md",
    ),
]

__all__ = ["PRESET_DOCS"]
