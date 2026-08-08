# 图像模态：Stable Diffusion、FLUX 与 DiT 简介

## 一、扩散模型（Diffusion Model）基本原理

扩散模型的灵感来自非平衡热力学，训练过程是对图片逐步加噪直到纯高斯噪声（前向扩散），学习一个网络去预测每一步的噪声从而逐步去噪还原图片（反向去噪）。

数学上，前向过程固定为马尔可夫链：q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)，其中 β_t 是固定的噪声调度。任意步 t 的带噪样本可直接由 x₀ 闭式采样得到：

    x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε,   ε ~ N(0, I),  ᾱ_t = Π_s=1..t (1-β_s)

训练目标为噪声预测 ε-预测（由 Ho et al. DDPM 2020 提出）：

    L_simple = E_{t, x₀, ε} [||ε - ε_θ(x_t, t)||²]

采样时从 x_T ~ N(0, I) 开始，按 t = T→1 逐步执行：x_{t-1} = 1/√α_t · (x_t - β_t / √(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z。DDPM 原始采样需 1000 步，后来 DDIM、DPM-Solver、Euler、Heun、LCM 等调度器将步数降至 20~50 甚至 1~8 步，极大提升了推理速度。

## 二、Stable Diffusion 架构详解

Stable Diffusion（CompVis / Runway / Stability AI, 2022）的核心创新是在"潜空间"做扩散，而非像素空间。整套管线包含 4 个组件：

1. **VAE（变分自编码器）**：将 512×512×3 的 RGB 图片编码到 64×64×4 的潜向量 z，空间下采样 8 倍。扩散在 z 上执行，生成完毕后解码回像素。VAE 的使用使 512×512 图的显存/计算量降低至原图的 1/48 左右，是 SD 能在消费级 GPU 上跑起来的关键。
2. **UNet 去噪网络**：标准 UNet 骨干，下采样 + 上采样 + 跳跃连接；中间层引入了 Cross-Attention 以注入文本条件。时间步 t 通过 Sinusoidal Embedding + MLP 注入到每层。
3. **CLIP 文本编码器**：通常使用 CLIP ViT-L/14（SD 1.x）或更大版本（SDXL 双编码器）。将输入 prompt 编码为 77×768 的文本嵌入，通过 Cross-Attention 喂给 UNet。
4. **Scheduler / Sampler**：负责调度噪声步，常见有 Euler a、DPM++ 2M Karras、UniPC、LCM-LoRA 等。

Classifier-Free Guidance（CFG）是高质量生成的核心：训练时随机置空 prompt 做无条件训练；采样时同时跑带条件 ε_c 与无条件 ε_u，最终梯度为：

    ε = ε_u + scale · (ε_c - ε_u)

scale（CFG Scale）一般 5~15，值越高生成越贴近 prompt，但过高会导致过饱和与伪影。此外 ControlNet、IP-Adapter、T2I-Adapter 等通过在 UNet 旁外挂条件网络，实现了边缘、深度、人体姿态、参考图、风格图等细粒度条件控制，极大拓展了 SD 的工业落地场景。

## 三、FLUX 架构：流匹配 + 最大扩散

FLUX.1（Black Forest Labs, 2024）是新一代开放权重文生图模型，在图像质量、语义对齐、人体结构、手部细节、图文忠实度上相对 SDXL 有显著代际提升。其核心特点：

1. **流匹配（Flow Matching, FM）目标**：不再使用传统 DDPM 的噪声预测，而是直接训练神经网络预测从纯噪声 z₀ 到数据 z₁ 的直线速度场 v：v_θ(z_t, t, cond) ≈ d z_t / dt。采样只需做 ODE 积分 z_t ← z_t + dt · v_θ，理论更简单、采样步数更少（FLUX 默认 20~50 步）。
2. **更大更现代的 Transformer 骨干**：放弃 UNet，全面转向 DiT（Diffusion Transformer）。将潜空间 patch 化为 token，标准 Transformer 层内部混合自注意力（self-attn）与文本-图像交叉注意力（cross-attn）。引入 Double Stream Blocks 分别处理图像 token 与文本 token，交互通过注意力完成。
3. **更大规模训练**：FLUX.1 [schnell] 是蒸馏 4 步采样版本，[dev] 是通用高质量版本，[pro] 为商业 API 版本。支持 1024×1024 及以上原生分辨率，通过 aspect-ratio 桶式训练在各种画幅上均表现优异。

## 四、DiT：Diffusion Transformer

DiT（Peebles & Xie, 2023）首次系统性地证明纯 Transformer 可以替代 UNet 作为扩散骨干，并且在类条件 ImageNet 生成上取得了 SOTA FID。核心结构：将 VAE 潜空间 patch 化，加入位置编码，送入标准 ViT 风格 Transformer Blocks。时间步与类别条件通过 Adaptive Layer Norm Zero（adaLN-Zero）注入到每个 Block，控制全局风格与语义。

后续的 PixArt-α、SD3、FLUX、Sora 等大型生成模型几乎全部采用 DiT 风格骨干，并在其中融合了多流处理、RoPE 2D 位置编码、RMSNorm、SwiGLU FFN 等 LLM 成熟技术，呈现出"大一统架构"的趋势：无论是语言、图像、视频，都在往"ViT + 条件注入 + 流匹配/扩散"的统一范式收敛，这为未来的多模态统一模型（如 sky-v1-omni 的 5-modal 架构）提供了清晰的技术路径。
