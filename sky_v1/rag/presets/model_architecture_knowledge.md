# Transformer 注意力机制、FlashAttention 与 LoRA 原理

## 一、Transformer 基础架构

Transformer 由 Google Brain 团队于 2017 年在《Attention Is All You Need》中提出，完全基于自注意力机制构建，摒弃了 RNN 的时序依赖，使序列任务可并行训练。

标准 Transformer 由编码器（Encoder）与解码器（Decoder）堆叠而成。每层包含两个核心子层：多头自注意力（Multi-Head Self-Attention, MHSA）与前馈网络（Feed-Forward Network, FFN），并通过残差连接与层归一化（LayerNorm）串联。FFN 通常为两层 MLP，中间维度是模型维度的 4 倍，激活函数使用 GELU 或 SwiGLU。

## 二、缩放点积注意力（Scaled Dot-Product Attention）

自注意力的数学表达如下：给定输入矩阵 Q（查询）、K（键）、V（值），维度为 d_model，通过线性投影得到 Q, K, V ∈ R^{n×d_k}，计算：

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

其中除以 √d_k 称为"缩放"，目的是抵消 d_k 增大时内积方差增长，避免 softmax 饱和进入梯度消失区。当序列长度 n 很大时，QK^T ∈ R^{n×n} 的显存占用与计算量呈 O(n²) 增长，这是 Transformer 在长序列上的核心瓶颈。

## 三、多头注意力（Multi-Head Attention, MHA）

MHA 将 Q, K, V 沿通道维度切分为 h 个头，每个头独立执行注意力，最后拼接回原维度：

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

多头机制使模型能在不同子空间捕捉不同模式的依赖关系：有的头建模语法依赖，有的头建模长距离指代消解，有的头捕捉局部 n-gram 模式。

## 四、FlashAttention 核心原理

FlashAttention（Dao et al., 2022）是对标准注意力的显存与计算双重优化实现，核心思想是分块计算（tiling）+ 重计算（recomputation），大幅减少 HBM 访问量。

标准注意力需要在 HBM 上构造完整的 n×n 注意力矩阵，FlashAttention 则将 Q、K、V 沿序列维切分成小块，每次仅加载一块 Q 与一块 K/V 到 SRAM 执行局部注意力，边计算边累加结果。反向传播时不保存完整的注意力矩阵，而是重新从 Q、K、V 与输出 O 推导梯度，避免了 O(n²) 显存。实测在 GPT-2 训练中可获得 2~4 倍加速，显存降低约 3 倍。FlashAttention-2、FlashAttention-3 进一步优化了 Warp 调度与 Ampere/Hopper 架构的 Tensor Core 利用率。

## 五、LoRA：低秩适配（Low-Rank Adaptation）

LoRA（Hu et al., 2021）是一种参数高效微调（PEFT）方法，其核心洞察是：大模型在下游任务上的权重更新本质上是低秩的。因此可冻结原始权重 W₀ ∈ R^{d×k}，仅训练两个低秩矩阵 B ∈ R^{d×r}、A ∈ R^{r×k}（r ≪ min(d, k)），前向传播时：

    h = W₀ x + (B A) x = W₀ x + ΔW x

其中 ΔW = BA 构成一个低秩更新。训练仅更新 A、B，参数量仅为原权重的 2rd/dk = 2r/k。推理时可将 BA 合并回 W₀，无额外延迟。LoRA 通常应用于自注意力的 Q、K、V、O 投影以及 MLP 的上下投影矩阵。常见的 r 取值为 8~64，alpha 与 r 的比值控制更新强度。与全量微调相比，LoRA 仅需 0.1%~1% 的参数即可达到接近全量微调的效果，并且能避免全量微调带来的灾难性遗忘。

## 六、相关技术扩展

注意力的变体包括稀疏注意力（Longformer、BigBird）、线性注意力（Linformer、Performer）、滑窗注意力（Sliding Window）与分块注意力（Block-local）。RoPE（旋转位置编码）通过对 Q、K 应用逐维旋转实现相对位置编码，被 LLaMA、Qwen、Mistral 等主流开源模型广泛采用。RMSNorm 作为 LayerNorm 的简化版本，仅做均方根归一化，减少了均值计算开销。SwiGLU 将 FFN 的 GELU 替换为 Swish 门控，在多数 LLM 中带来了稳定的困惑度提升。
