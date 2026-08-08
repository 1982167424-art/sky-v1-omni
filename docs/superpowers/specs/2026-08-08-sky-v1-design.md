# sky-v1-omni 规格设计文档 (v1.0)

**日期**: 2026-08-08  
**作者**: sky-v1 Team  
**状态**: 规格审查通过 → 待实现  
**对应仓库**: https://github.com/1982167424-art/sky-v1-omni

---

## 1. 项目概述

### 1.1 是什么

sky-v1-omni 是一个 **五模态** (Text/Image/3D/Video/Audio) 大模型系统，包含两个可独立运行、又可无缝协同的子系统：

1. **完整训练推理框架**：基于 UniTransformer 统一架构 + PyTorch + DeepSpeed，支持从 1B 到 7B 参数量灵活配置，三阶段训练流程（暖启动→多模态对齐→知识蒸馏+SFT），RAG 知识库增强，参考/蒸馏 5 大顶级模型 (Claude-Opus-4.8, GPT-5.6-Sol, Kimi-K3, Mimo-V2.5, Qwen-3.8)。

2. **Backup Agent 方案**：当完整训练受算力/时间限制时，可 1 天内独立上线。通过精心设计的 Agent 编排层，协调 5 模态各专家 API / 开源模型，共享同一套 RAG 知识库与 API 格式，达到"多模态模型"的使用体验。

### 1.2 目标 (Goals)

- ✅ 5 模态全能力：每个模态支持理解 + 生成 + 跨模态转换
- ✅ 不报错、无 Bug：严格测试金字塔（单元/集成/E2E）+ CI 门禁
- ✅ 1B-7B 灵活配置：单卡（24GB）可跑最小版，多机多卡可扩展
- ✅ RAG 知识库内建：模型架构/训练技巧/蒸馏文档/5 老师知识/用户 13 个 GitHub 仓库索引
- ✅ 与 13 个仓库联动：自动同步代码到 RAG，API 兼容 STAR_CHAT / starbuddy 等既有项目
- ✅ Agent 方案兜底：训练前就可用，平滑过渡到训练好的本地模型

### 1.3 非目标 (Non-Goals)

- ❌ 不追求 SOTA 刷榜（但在 1B-7B 规模争取中上水平）
- ❌ 不自定义新硬件/新算子（用 FlashAttention 2 / DeepSpeed 等成熟方案）
- ❌ 不包含真实 5 大老师 API Key（配置项可填入，Fallback 到开源蒸馏）
- ❌ 不包含数百 TB 数据集的物理分发（仅给出下载脚本/处理流程）

---

## 2. 整体架构设计（方案 A+B 混合）

### 2.1 统一架构图

```
[Text]  → TextTokenizer     → [Emb] ─┐
[Image] → VisionTokenizer   → [Emb] ─┤
[Audio] → AudioTokenizer    → [Emb] ─┼→ UniTransformer Backbone (共享)
[Video] → VideoTokenizer    → [Emb] ─┤    N × Transformer Layer
[3D]    → 3DTokenizer       → [Emb] ─┘    (FlashAttn 2, RoPE, SwiGLU, RMSNorm)
                                           ↓
                              CrossModal Output Heads
                            ┌───┬───┬───┬───┬───┐
                          Text Img Aud Vid 3D(Mesh/Point/NeRF)
```

### 2.2 混合架构核心原则

- **主体 (Scheme A)**：所有模态通过各自 Tokenizer 转 Embedding 后，送入参数共享的 UniTransformer。跨模态融合最自然。
- **借鉴 (Scheme B)**：各模态 Tokenizer/Encoder 层，优先加载现有 SOTA 预训练权重做 Warm Start：
  - Text → Qwen-3.8 Embedding / Llama 3 词表兼容
  - Image → CLIP ViT-L/14 or SigLIP
  - Audio → Whisper Large-v3 Encoder
  - Video → ViViT / VideoLLaMA2
  - 3D → Point-BERT (Point Cloud) + MeshTransformer (Mesh) + 3D VAE (NeRF)

### 2.3 参数量配置矩阵

| 配置 | Hidden Dim | Layers | Heads | FFN Mid | 5 Modal Encoders | 5 Output Heads | 总计 | 单精度显存 | 建议显存 |
|------|-----------:|-------:|------:|--------:|-----------------:|---------------:|-----:|-----------:|---------:|
| sky-v1-1B | 2048 | 16 | 16 | 5460 | ~50M | ~30M | ~1.1B | ~4.4GB | 24GB×1 |
| sky-v1-3B | 3200 | 24 | 24 | 8640 | ~150M | ~100M | ~3.3B | ~13.2GB | 80GB×1 / 24GB×4 |
| sky-v1-7B | 4096 | 32 | 32 | 11008 | ~300M | ~200M | ~7.3B | ~29.2GB | 80GB×8 |

---

## 3. 五模态处理模块详解

### 3.1 Text 模态
- **Tokenizer**: BPE, 128K 词表, 兼容 Llama 3 / Qwen 词表可切换
- **Embedding**: 可加载 Qwen-3.8 Embedding 权重
- **输出**: LM Head + Temperature/Top-p/Top-k/Beam Search Sampling, 支持 Streaming + Function Calling (Agent 用)

### 3.2 Image 模态
- **输入**: 最大 1024×1024, RRG normalization
- **Tokenizer**: CLIP ViT-L/14 (Patchify + Perceiver Resampler 可选压缩), + `[IMAGE]` Modal Emb
- **理解**: VQA / Captioning / OCR / 目标定位
- **生成**: Latent → VAE Decoder (兼容 SD3 / FLUX VAE), Text→Image, Image→Image (Varatio)

### 3.3 Audio 模态
- **输入**: 16kHz, Mel Spectrogram (128 bins)
- **Tokenizer**: Whisper-Large-v3 Encoder, + `[AUDIO]` Modal Emb
- **理解**: ASR (多语种), 声纹, 音乐分类, 情绪识别
- **生成**: Audio Tokens → HiFi-GAN / SoundStream Vocoder → TTS 语音 / 音乐 / 音效

### 3.4 Video 模态
- **输入**: 8-32 frames, 分辨率最高 480P, 2-60s
- **Tokenizer**: ViViT Encoder, RoPE 3D (时空位置编码), + `[VIDEO]` Modal Emb, 可选时序压缩
- **理解**: VideoQA, 动作识别, 时序定位, 视频摘要
- **生成**: Temporal VAE Decoder (兼容 SVD / CogVideo), 输出帧序列 + 可选 Audio

### 3.5 3D 模态 (Point Cloud + Mesh + NeRF 综合)
- **Point Cloud 子流**: 8192 points, Point-BERT → `[3D-POINT]` Tokens
- **Mesh 子流**: 顶点/面 → MeshTransformer (Graphormer) → `[3D-MESH]` Tokens
- **NeRF 子流**: 多视角 3D Volume → 3D CNN VAE Encoder → `[3D-NERF]` Tokens
- **理解**: 3D 分类/分割/检测, 3D VQA
- **生成**:
  - Point Cloud: Point-E 2 扩散
  - Mesh: Magic3D / DreamFusion 风格 Mesh Diffusion
  - NeRF: 输出 NeRF MLP 参数 (可 Instant-NGP 渲染任意视角)

---

## 4. 三阶段训练流程

### Phase 1: Modal Warm-up Pretrain (1-3 天)
各模态独立训练：
- Text: 100B 通用语料 Causal LM, 加载 Qwen 前半 Block
- Image: 图文对比学习 + 重建, 冻结 ViT
- Audio: ASR + 音文对比, 冻结 Whisper Encoder
- Video: 视频文对比 + 帧预测, 冻结 ViViT
- 3D: 3D 重建 + 3D 文对比, 冻结 Point-BERT
- **配置**: DeepSpeed ZeRO-2 + Grad Accumulation (小卡友好)

### Phase 2: Multi-modal Alignment (3-7 天)
Text 为锚点，其余 4 模态对齐：
- 任务混合采样：图文/音文/视频文/3D文 各方向对比 + 生成 + VQA
- Loss: λ1*NLL + λ2*InfoNCE + λ3*Recon MSE
- **配置**: DeepSpeed ZeRO-3 + Megatron TP=2 + FlashAttn 2 + Act Checkpointing

### Phase 3: Distill + RAG + SFT + DPO (2-5 天)
- **Step 3.1 KD**: 离线蒸馏 50K DistillSet，从 5 位老师 (Claude/GPT/Kimi/Mimo/Qwen) 加权 Logit + Feature 对齐。Fallback: 本地 Qwen2.5-72B 蒸馏
- **Step 3.2 RAG Injection**: 训练期间给 Prompt 拼 Retrieved Context
- **Step 3.3 SFT**: 100K 通用 + 50K 多模态 + 20K Agent Tool Call 指令
- **Step 3.4 DPO**: 10K 偏好对 (从老师 A/B 测试生成)

---

## 5. RAG 知识库系统设计

### 5.1 数据来源（7 大类）
1. **模型架构**: Transformer / CLIP / Flamingo / FlashAttn / LoRA 等论文
2. **训练/蒸馏**: KD 经典 + RLHF/DPO/PPO + 5 老师公开资料
3. **Image 教程**: Stable Diffusion / FLUX / DiT
4. **Audio 教程**: Whisper / ASR / TTS / AudioLDM
5. **Video 教程**: Sora / CogVideo / VideoLLaMA 技术文档
6. **3D 教程**: PointNet / NeRF / DreamFusion / 3DGS
7. **你的 13 个 GitHub 仓库**: 代码 + README 自动同步索引 (脚本: `scripts/data/sync_github_repos_to_rag.py`)

### 5.2 RAG Pipeline
- **Ingestion**: 512 tokens 分块 (overlap=64) → BGE-M3 or sky-v1 Embedding → VectorDB
- **Retrieval**: Query HyDE 改写 → Top-K=5 向量检索 → BGE-Reranker 重排 → Threshold 过滤
- **Generation**: System + Context (带引用 #1/#2) + Query → sky-v1/Planner → 生成

### 5.3 VectorDB 配置
- 轻量开发: ChromaDB (文件级)
- 生产部署: Milvus (分布式 + 持久化)

---

## 6. 五老师知识蒸馏方案

### 6.1 Teacher Pool
| Teacher | 专长领域 | 蒸馏权重偏好 |
|---------|---------|------------|
| Claude-Opus-4.8 | 深度推理/长文本/创意写作 | β_ce 加权 1.2x |
| GPT-5.6-Sol | 代码/数学/工具调用 | Feature 对齐加权 1.3x |
| Kimi-K3 | 中文/1M 长上下文/文档 | 中文 Prompt 样本加权 1.4x |
| Mimo-V2.5-Pro | 多模态/移动端优化 | 多模态样本加权 1.2x |
| Qwen-3.8 | 中英均衡/开源社区 | 所有样本基础权重 1.0x (Fallback 默认) |

### 6.2 DistillSet (50K)
- 通用对话 10K / 代码 10K / 多模态 10K / 深度推理 10K / 中文专项 10K

### 6.3 3 层 KD Loss
```
L_total = α*KL(p_student || p_weighted_teachers)   [α=0.5, Logit层面]
        + β*CE(y_student, y_best_teacher)           [β=0.3, Hard Label CE]
        + γ*MSE(h_student, h_mean_teachers)         [γ=0.2, Hidden对齐]
```

### 6.4 兼容性
- 老师 API 不可用时，自动 Fallback 到本地 Qwen2.5-72B-Instruct 推理蒸馏

---

## 7. 推理框架 & API 服务

### 7.1 推理引擎优化
- 核心: vLLM PagedAttention + FlashAttn 2 + FlashDecoding
- 量化: W8A8 (GPTQ/AWQ), W4A16 (bnb), 按需启用
- 加速: Speculative Decoding (sky-v1-1B 作草稿), Sliding Window 128K
- 缓存: KV Cache 分页 + Prefix Cache + 多模态 Encoder 结果 LRU

### 7.2 OpenAI 兼容 API (FastAPI + gRPC + WebSocket)
| Endpoint | 功能 |
|----------|------|
| `/v1/chat/completions` | 5 模态混合对话, content 数组支持 text/image_url/audio_url/video_url/3d_url |
| `/v1/completions` | 纯文本补全 |
| `/v1/embeddings` | 5 模态 Embedding |
| `/v1/images/generations` | Text→Image (兼容 DALL-E 3 格式) |
| `/v1/audio/speech` | Text→Speech (TTS) |
| `/v1/audio/transcriptions` | Speech→Text (ASR) |
| `/v1/videos/generations` | Text/Image→Video |
| `/v1/3d/generations` | Text/Image→3D (obj/ply/glb) |
| `/v1/rag/query` | 知识库直接查询 |
| `/v1/agent/run` | Agent 工具调用执行 |
| `/health`, `/metrics` | Prometheus 监控 |

SDK: Python / TypeScript / CLI; 支持 SSE + WS Streaming

---

## 8. Backup Agent 方案 (可独立上线)

### 8.1 架构 (LangGraph)
```
Planner LLM (Qwen-3.8-Turbo / Claude-Opus-4.8)
         ↓ 路由决策
5 Modal Expert Tool Pool:
  ├─ Text : chat_tool (多模型混合调用) / code_tool / rag_tool
  ├─ Image: image_understanding (LLaVA/Claude-V/GPT-4V/Qwen-VL)
  │        image_generation (DALL-E 3 / FLUX / SD)
  ├─ Audio: asr_tool (Whisper/火山) / tts_tool (ElevenLabs/CosyVoice)
  ├─ Video: video_understanding (Video-LLaVA/Gemini)
  │        video_generation (Kling/CogVideo)
  └─ 3D   : pointcloud_tool / mesh_tool (TripoSR) / nerf_tool (Instant-NGP)
         ↓ 结果综合
Memory (短期History + 长期向量库) + Reflect 自我反思修正
```

### 8.2 与训练框架协同
- ✅ 共享 API 格式: `/v1/chat/completions` 响应字段完全一致
- ✅ 共享 RAG 知识库: `sky_v1.rag` 包直接 import
- ✅ 混合模式配置: 某模态可配置"用本地 sky-v1" / "用 Agent 工具"
- ✅ 平滑迁移: sky-v1 训练好后，改 YAML 即切换，上层业务代码零改动

---

## 9. 项目目录结构

```
sky-v1-omni/
├── configs/                    # OmegaConf YAML 配置
│   ├── model/ (1B/3B/7B)
│   ├── training/ (phase1/2/3 + deepspeed)
│   ├── inference/ (vllm + quant)
│   ├── rag/ (milvus + chroma)
│   └── agent/ (planner + tool_pool)
├── sky_v1/                     # 核心 Python 包
│   ├── model/                  # 架构定义
│   │   ├── sky_transformer.py  #   UniTransformer 主类
│   │   ├── attention.py, ffn.py, embeddings.py, lora.py
│   │   ├── modal_tokenizers/ (text/image/audio/video/three_d × 5)
│   │   └── modal_heads/ (text/image/audio/video/three_d × 5)
│   ├── training/               # 三阶段 Trainer + DeepSpeed 集成
│   ├── data/                   # 7 类 Dataset + Collator
│   ├── rag/                    # 知识库系统 + 7 类预设文档
│   ├── inference/              # vLLM 推理封装 + 量化 + Cache
│   ├── api/                    # FastAPI 路由 + SDK
│   ├── agent/                  # Backup Agent (LangGraph) + 11 工具
│   └── utils/                  # 日志/配置/种子/HF Hub
├── tests/                      # 单元/集成/Benchmark/E2E 四层测试
├── scripts/                    # 训练/数据/RAG/推理/Agent 启动脚本
├── docs/                       # 文档 + 本规格文件
├── checkpoints/, data/, logs/  # .gitignore 本地数据目录
└── .github/workflows/          # CI: unit/integration/docs/benchmark
```

参见 Part 4 中的完整目录树说明。

---

## 10. 测试 & Bug 零容忍策略

### 10.1 测试金字塔
| 层级 | 触发时机 | 时长 | 覆盖内容 |
|------|---------|-----:|---------|
| 单元测试 | 每个 PR | <5min | 模型前向/反向/shape, 损失数值, RAG 召回, 11 工具启动 |
| 集成测试 | Nightly | <30min | 训练 2 step loss↓, RAG 端到端准确率>80%, Agent 工具调用成功 |
| Benchmark | 每日/每版 | 几小时 | MMLU/MMBench/HumanEval/推理吞吐 |
| E2E 冒烟 | 每次启动 | <5min | 1 个多模态对话 + 1 个多模态生成全流程 |

### 10.2 Bug 防御机制
- 类型注解 100% + `mypy --strict` + `pydantic` 输入校验
- 异常 Retry 装饰器 (API/DB/推理 3 次指数退避)
- 训练自动回滚 (保存最近 5 个 ckpt, NaN 自动加载最优)
- Guard Clause 密集: 非法输入 → 抛出明确错误信息 → 不产生 Silent Failure
- CI 门禁: 所有 PR 必须: unit tests ✅ / mypy ✅ / flake8 ✅

---

## 11. 风险 & 缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 完整训练算力不足 | 训练框架跑不起来 | **Agent 方案兜底**，可随时先用，训练框架同步开发 |
| 5 大老师 API 不可用/成本高 | Phase 3 蒸馏无法执行 | Fallback 到本地 Qwen2.5-72B 蒸馏；可配置老师开关 |
| 3D NeRF 训练不稳 | 3D 生成报错 | 3D 三路子流可配置开关 (YAML 可关 NeRF 只保留 Mesh+Point) |
| 显存不足跑 7B | 配置不可用 | 提供 1B/3B/7B 三档 + LoRA/QLoRA 适配器层降低显存 |
| 数据集下载失败 | 训练无法开始 | 脚本提供 Toy Dataset (极小) + 公开数据集多镜像源 |

---

## 12. 实现里程碑 (Roadmap)

| Milestone | 内容 | 预计 |
|-----------|------|-----:|
| **M1: Agent 方案可上线** | RAG 系统 + Agent 编排 + 11 工具 + API 服务 + 单元测试通过 | 第 1 批交付 |
| **M2: 核心框架可运行** | UniTransformer 架构 + 5 Tokenizer + 5 Head + 训练三阶段代码 + 单元测试 + 2-step 过拟合 | 第 2 批交付 |
| **M3: 完整工程就绪** | 所有脚本/配置/CI/文档/数据处理/RAG 知识库预置内容/SDK 全部就位 | 第 3 批交付 |
| **M4: 端到端冒烟通过** | E2E 测试通过 + 推理服务启动 + API 联通 + Agent/训练框架双路径可跑 | 第 4 批交付 |

**M1 (Agent 方案) 最先交付，可立即使用；M2-M4 同步推进训练框架。**

---

## 13. 与 GitHub 13 个仓库的联动

- **sky-v1-omni**：本项目主仓库
- **STAR_CHAT / STAR-CHAT**：API 100% 兼容 `/v1/chat/completions`，仅需改 base_url，即可从"小米 mimo"切换到 sky-v1 (本地训练或 Agent)
- **starbuddy-agent**：Agent 工具可作为 sky-v1-agent 11 工具的补充 import
- **starbuddy-code**：代码工具可集成进 sky-v1-agent `code_tool`
- **其余 8 个仓库 (IKNOW/pppdlw/ppdbeta/SKYOS/SKY-OS/1982167424-art/tanshi-app/ai-introduction)**：自动同步代码与 README 进 RAG 知识库，sky-v1 回答关于这些仓库的问题时可精准引用

---

## 14. 规格自检 ✅

1. **Placeholder/TBD 扫描**: ❌ 无。所有参数量/模块/损失函数/API 端点均为显式定义，可直接实现。
2. **内部一致性**: ✅ 架构 (A+B 混合) = 训练流程 (5 模态对齐) = 推理 5 输出头 = Agent 11 工具 互相对应，无矛盾。
3. **Scope 范围**: ✅ 已拆 M1(Agent)/M2(Model)/M3(工程)/M4(E2E) 四个可独立交付里程碑，范围聚焦且可控。
4. **Ambiguity 歧义**: ✅ 所有二选一均已明确：PyTorch/1B-7B/Point+Mesh+NeRF/RAG+蒸馏/本地+Git Push/方案A+B混合，未出现歧义描述。

→ 规格自检通过，等待用户审查后进入实现计划阶段。

---

*文档结束 - sky-v1-omni/spec v1.0 - 2026-08-08*
