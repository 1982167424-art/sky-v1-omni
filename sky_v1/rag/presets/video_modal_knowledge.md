# 视频模态：Sora、CogVideo、VideoLLaMA 简介

## 一、视频生成的独特挑战

视频本质是"时序连贯的图像序列 + 音频"，相比单张图像生成有 3 个核心难点：
1. **时空双维度一致性**：不仅每帧本身要真实，相邻帧之间的物体、相机运动、光照、遮挡必须物理一致，否则出现"闪烁"、"穿模"、"物体突变"。
2. **算力与显存爆炸**：一段 10 秒 1080p 30fps 视频 = 300 帧 ≈ 100~1000 张高清图的生成量，显存与计算量是图生的 1~2 个数量级。
3. **长时序语义对齐**：用户一句 prompt 描述 30 秒故事，模型需要全局规划镜头、角色、事件因果，不能前后矛盾。

因此视频模型架构通常在图像 DiT 基础上增加"时间注意力（Temporal Attention）"、"3D Patch 化"、"时空因子化分解"等设计，通过时空解耦、级联生成、VAE 时空压缩、流匹配加速等手段把成本压到可接受范围。

## 二、Sora：大规模视频 DiT 的里程碑

Sora（OpenAI 2024）是第一个在生成质量、时序一致性、物理因果、时长（最长 60 秒 1080p）上全面超过传统 GAN/扩散视频模型的产品级方案。其架构与训练范式的核心要点：

### 2.1 视觉"补丁化"（Visual Tokens / Patches）统一范式
Sora 借鉴 LLM 的思想：无论什么分辨率、时长、画幅的视频或图片，先通过"时空 VAE"（Recurrent Video Compression Network）压缩成统一格式的离散/连续潜空间 patch token，再送入一个标准 Transformer（DiT 风格）做下一 patch 预测或流匹配目标。这种统一表示使得同一模型能无缝处理图像、视频、竖屏、横屏、方屏，且能"续接"生成更长内容。

### 2.2 时空 Transformer Block（Spacetime DiT）
Sora 内部的 Transformer 层使用时空解耦注意力：对空间维度（同一时刻所有 patch）做自注意力，对时间维度（同一空间位置沿时间轴）做滑动时间窗自注意力，交叉注意力从文本 CLIP/T5 嵌入读取条件。还引入了 **3D RoPE（三维旋转位置编码）** 对 H/W/T 三维分别编码，使模型理解时空相对关系，保证长视频的连贯性。

### 2.3 训练数据与质量
Sora 使用大规模真实视频 + 合成数据混合训练，通过字幕模型（类似 Dense Video Caption）给视频生成高细粒度逐段描述，保证图文对齐。训练还包含"视频剪辑拼接"、"视频扩展补全（Video Inpainting）"、"图生视频"、"视频到视频"等多任务联合，让模型具备多种编辑能力。学术界普遍认为其模型规模在 40B~80B 级别参数，训练数据达到千万小时级。

### 2.4 当前局限
Sora 在复杂物理交互（如手精确抓取物体）、长时序因果一致性（A 场景出现后 B 场景应该出现的关联物品可能丢失）、真实世界物理规律（液体飞溅、布料撕裂）仍有瑕疵。未来方向是与世界模型（World Model）、可微物理引擎、多视角 3D 监督深度融合。

## 三、CogVideo：开源多阶段级联视频生成

CogVideo（智谱 AI 2022-2024）是开源可复现视频生成的代表系列，CogVideoX-5B 在开源 Sora 级模型中第一梯队。采用经典三阶段级联（Cascade）架构，通过"先低分辨率粗 → 再高分辨率精 → 最后补帧到高帧率"解耦难度：

### 3.1 阶段 1：文生 480P 稀疏关键帧
输入 T5-XXL 编码的文本条件，Video DiT 生成 480P、T=13 或 T=21 稀疏关键帧的低帧视频。该阶段只负责"语义正确 + 画面构图 + 大致运动"。

### 3.2 阶段 2：空域超分（Spatial Super-Resolution）
把 480P × T_稀疏 送入空间 Video SR DiT，上采样到 720P 或 1080P，补全纹理与细节，时序帧数不变。

### 3.3 阶段 3：时域插帧（Temporal Interpolation）
把低帧率视频送入 Temporal Interpolation DiT，在相邻关键帧间预测中间帧，把帧率从 4fps→8fps→16fps→32fps 逐级补到流畅，最终输出 6~10 秒的 720P/1080P 24~30fps 视频。

### 3.4 CogVideoX 关键设计
- **3D VAE**：像 Stable Diffusion 那样同时对 H/W 做 8× 下采样、对 T 做 4× 下采样，极大降低中间表示大小；
- **ExpertFFN / MoE 化**：稀疏激活 MoE FFN 层，在不增加算力前提下扩大有效参数；
- **Plan-Then-Generate 指令分解**：对长文本 prompt 先用 LLM 拆解为"镜头 1 描述 / 镜头 2 描述 / ..."时序脚本，再逐段生成后拼接，增强长视频因果；
- **多条件融合**：支持 Text-to-Video、Image-to-Video（参考首帧）、Video-to-Video（风格迁移/动作迁移）三种输入条件组合。

## 四、VideoLLaMA：视频理解 + 大语言模型多模态对话

VideoLLaMA / VideoLLaMA 2 / Video-LLaVA 系列工作属于"视频理解模型"，目标不是生成视频而是"看懂视频并用自然语言回答问题"。这类模型的核心是"视频编码器 → 投影对齐 → 冻结/可微调 LLM"三部分：

### 4.1 视频时空编码器
从海量视频预训练得到的 Video ViT 中提取特征。常用的有：
- **EVL / InternVideo**：百万级视频对比学习预训练的 VideoMAE 风格 ViT；
- **TimeSformer / ViViT**：时空分离/联合注意力的 Video Transformer；
- **CLIP ViT-L/14 逐帧 + Temporal Pooling**：简单高效，很多基线用。

视频 V→H×W×T×C 先均匀抽帧（T=8/16/32），每帧独立送入图像 ViT，得到每帧的 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 或 patch token。之后用一个 Temporal Transformer 把帧级特征沿时序维度做自注意力，得到全局视频表示，或者保留时空 token 直接送入 LLM（这需要 LLM 侧支持更大的上下文）。

### 4.2 Q-Former / Projection：视觉到语言的对齐桥梁
视觉特征维度与语言词嵌入维度不一致，且语义空间不同。需要用 **Q-Former（BLIP-2 提出）** 或 **线性投影 / MLP 投影层** 将视频特征"翻译"成 LLM 能理解的伪 token 序列：

    v_tokens = Proj(VideoEncoder(video))  # 形状 (N, D_LLM)

推理时把这些伪 token 插入到对话 prompt 的对应位置，LLM 像读文字一样"读"视频表示，然后生成回答。训练分两阶段：第一阶段冻结 LLM 与视频编码器，仅训练 Proj/Q-Former（图文对比 + 生成对齐）；第二阶段解冻 LLM 用 LoRA，在视频 QA 数据集上做指令微调（Video-ChatGPT、MVBench、Video-MME 等）。

### 4.3 典型任务与评测
- **开放域视频问答（Open-ended Video QA）**："视频里第一个打开门的人穿什么衣服？"
- **时序定位 + QA**："什么时候主角拿到了钥匙？用秒级时间戳回答"
- **视频字幕（Dense Caption）**：逐秒生成视频描述
- **多视频对比 QA**：给两段视频问差异

评测基准常用 MSR-VTT、MSVD、TGIF、ActivityNet-QA、NExT-QA、Perception Test、Video-MME 等。目前开源 VideoLLaMA 类模型在 7B~13B 规模上已能处理日常视频理解，与商业 GPT-4V(ision) / Gemini Advanced 的差距主要集中在细粒度时序推理、物理常识、多人交互的长视频追踪上。

## 五、视频模态与 sky-v1-omni 五模态融合的展望

sky-v1-omni 作为 5-modal Omni Model，视频模态承担了"理解动态世界"与"生成动态内容"两大职责。未来可将 VideoLLaMA 风格的理解头、CogVideo 风格的级联生成头与 Sora 风格的统一 Visual Tokenizer 三者整合到统一架构：共享时空 VAE 与共享 Video DiT 骨干，通过任务切换头（Task Routing）切换"理解/生成"、"短/长"、"低/高分辨率"模式，配合文本、音频、图像、3D 的跨模态交叉注意力，实现用户一句"把这个产品三维模型(3D)配上我上传的 bgm(音频)，做一段 15 秒的电商宣传片，旁白用中文女声(TTS)，最后加上字幕"这种真正端到端的多模态创作。
