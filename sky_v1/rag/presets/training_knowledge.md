# 大模型三阶段训练流程简介

## 一、预训练（Pre-training）：从无到有的语言建模

预训练是大模型获得通用语言能力的基石阶段。训练数据通常是从互联网、书籍、百科、代码、论文等多源收集的数万亿 token 的大规模文本语料。数据清洗流程包括：语言过滤、去重（MinHash LSH）、低质量过滤（基于困惑度分类器）、去重 URL/Domain、敏感词过滤、质量分层（通常按"教科书级 → 网页 → 社交"分层加权采样）等。

目标函数是自回归语言模型（Auto-Regressive LM）的负对数似然：给定上下文 token x₁..x_t，预测下一个 token x_{t+1} 的交叉熵损失：

    L_pretrain = -Σ log P(x_{t+1} | x_1, ..., x_t)

训练使用分布式混合精度（BF16/FP16），优化器为 AdamW（β₁=0.9, β₂=0.95, weight_decay=0.1），学习率预热后余弦衰减。全局批大小通常在百万 token 量级。模型结构在 7B 以下常用 32~40 层、4096 上下文、32 头；70B 以上则需要 80 层以上并使用张量并行 + 流水线并行（3D 并行）。训练中常见的稳定性技巧包括：梯度裁剪（norm 1.0）、loss 缩放、RMSNorm、SwiGLU、RoPE、分组查询注意力（GQA）等。

## 二、监督微调（Supervised Fine-Tuning, SFT）：学习指令遵循能力

预训练模型学会了"续写"，但尚未学会"遵循人类指令回答"。SFT 阶段使用高质量的（指令、回复）pair 数据对模型继续微调，使其学会理解并执行用户指令。

SFT 数据来源包括：开源指令数据集（Alpaca、ShareGPT、Orca、Flan 等）、人工标注、GPT-4 等强模型自指令蒸馏（self-instruct）。数据需要严格去重、去对抗、去敏感信息。训练策略通常是：冻结 embeddings 和 LN，学习率降至预训练的 1/10~1/100，训练 1~3 epoch，防止过拟合。损失函数仍然是下一个 token 的交叉熵，但 mask 掉 prompt 部分的 loss，仅在回复段计算梯度（即 "response-only loss"）。

SFT 常见的对话模板有 ChatML、Llama 2 Chat Template、Alpaca 模板等，必须在训练与推理时保持严格一致，否则模型会出现"不按预期格式回复"的问题。

## 三、对齐（Alignment）：从有用到无害的最后一公里

RLHF（Reinforcement Learning from Human Feedback）是当前主流的对齐方法，分为三步：

### 3.1 人工排序数据收集
对相同 prompt 生成多个（K=4~7）回复，由标注者按偏好从优到劣排序，得到两两比较（pairwise preference）数据。

### 3.2 奖励模型（Reward Model, RM）训练
将 SFT 模型的最后一层去掉替换为标量头（reward head），输入 prompt+response，输出标量奖励 r。损失使用 Bradley-Terry 排序损失：

    L_RM = -E_{(y_w, y_l)~D} log σ(r(x, y_w) - r(x, y_l))

其中 y_w 是更好的回复，y_l 是更差的回复。好的 RM 需要在 OOD 测试集上有高准确率（>65%），并对"奖励黑客"（reward hacking）有鲁棒性。

### 3.3 PPO / DPO 强化学习
PPO（Proximal Policy Optimization）将 SFT 模型作为策略 π，RM 作为奖励信号，训练目标是：

    L_PPO = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) A)] - β KL(π || π_ref)

其中 A 是 GAE 估计的优势函数，KL 惩罚防止策略偏离 SFT 参考模型太远。

DPO（Direct Preference Optimization, Rafailov et al., 2023）绕过了显式 RM 与 PPO，直接把偏好对转化为对策略的分类损失：

    L_DPO = -E log σ(β [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

DPO 实现简单、稳定、显存占用低，已在 Zephyr、Qwen、Llama 3 等多个开源模型上取得与 RLHF 相当甚至更好的对齐效果，是当前开源社区的首选对齐方案。

三阶段的质量比例大致为：预训练 90%（通用能力地基）、SFT 8%（指令遵循接口）、对齐 2%（无害 + 有用的态度打磨）。三阶段缺一不可，跳过任何一步都会显著影响最终体验。
