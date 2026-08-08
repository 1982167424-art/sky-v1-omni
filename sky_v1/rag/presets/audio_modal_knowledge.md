# 音频模态：Whisper ASR、TTS 与 HiFi-GAN 简介

## 一、语音信号基础与特征表示

数字语音是对连续声波按采样率（8 kHz 电话、16 kHz 宽带、22.05/24/44.1/48 kHz 高保真）采样得到的一维时间序列。波形（Waveform）直接训练模型对时序建模能力要求高，因此 ASR 与 TTS 通常先转换为紧凑的声学特征：

- **梅尔频率倒谱系数（MFCC）**：对波形做 STFT → 取功率谱 → 通过梅尔滤波器组 → 取 log → DCT，得到 13~40 维特征，经典 ASR（GMM-HMM）时代的标配。
- **梅尔谱图（Log-Mel Spectrogram）**：仅到梅尔 log 功率为止，保留了频率轴维度，典型形状为 80 或 128 维 × 帧数，是现代端到端 ASR / TTS 与扩散声码器最常用的中间表示。
- **F0 / 音高曲线**：基频轨迹，用于多说话人、情感、歌唱语音合成。
- **HuBERT / Wav2Vec 2.0 自监督表征**：通过大量无标注音频预训练得到的深层隐藏表示，在低资源 ASR 上效果显著。

## 二、Whisper：大规模弱监督语音识别

Whisper（Radford et al., OpenAI 2022）是里程碑级的通用 ASR 模型，通过 68 万小时弱监督（网页转录字幕）多语言多任务训练，获得了极强的鲁棒性与零样本跨语种能力。

### 2.1 架构：Encoder-Decoder Transformer
- **音频编码器**：80 维 Log-Mel 特征（每 10ms 一帧，25ms 窗），经 2 层步长 2 的 Conv1D 降采样到每帧对应 30ms，位置编码使用可学习正弦编码，随后通过 32~96 层（Whisper tiny → large-v3）Transformer Encoder 得到 1500 帧的音频上下文化表征。
- **文本解码器**：自回归 Decoder 接收特殊 token 序列（<|startoftranscript|>, <|zh|>, <|transcribe|>, <|notimestamps|>...），通过 Cross-Attention 读取音频编码，输出识别文本 + 可选时间戳 token。

### 2.2 多任务联合训练
Whisper 一次前向同时解决 5 个子任务：语音识别（转录/翻译）、语种检测（X→en 翻译 / X→X 转录）、时间戳对齐、断句、说话人切分（在最新 whisper-v3 / whisper-large-v3-turbo 上加强）。训练数据覆盖 97 种语言，零样本可直接做任意语言的 ASR 与 X→en 翻译，在噪声、口音、专业词汇上比传统 HMM-DNN 方案鲁棒得多。

### 2.3 Whisper 家族与部署
按参数量：tiny (39M) → base (74M) → small (244M) → medium (769M) → large-v3 (1.55B)。large-v3 在中文 WER 约 5~8%，达到实用级。Whisper.cpp（C++ 移植）可在手机、树莓派甚至浏览器 WASM 上实时推理；faster-whisper 基于 CTranslate2 量化，在 GPU 上比原生快 4 倍、显存低 3 倍。Distil-Whisper 通过知识蒸馏在 512M 参数量上逼近 large-v3。

## 三、TTS：从文本到语音的主流范式

现代 TTS 通常分为两个阶段：
1. **前端/声学模型**：文本 → 梅尔谱；
2. **声码器（Vocoder）**：梅尔谱 → 波形。

### 3.1 Tacotron 2 声学模型
Tacotron 2（Google 2017）是两阶段范式的奠基工作：CBHG 文本编码器 → 注意力 LSTM 解码器（自回归逐帧输出 80 维 Mel）→ 后处理 Post-Net 修正残差。注意力机制对齐文本字符与声学帧，解决可变长度输入输出的对齐问题。

### 3.2 FastSpeech / FastSpeech 2：非自回归提速
自回归解码 1 秒音频需跑 100 步，太慢。FastSpeech（微软 2019）引入 **时长预测器（Duration Predictor）**：先用训练好的自回归教师模型（Tacotron 2）提取每个音素的帧数，训练一个小型 MLP 直接预测每段时长；推理时按时长把音素表征复制扩展为帧级，然后一次非自回归前向输出全部 Mel，速度提升 200 倍以上。FastSpeech 2 进一步加入 F0 预测器、能量预测器、多说话人嵌入，使合成音质与韵律接近甚至在某些指标上超过 Tacotron 2。

### 3.3 VITS：端到端单阶段 TTS
VITS（Kim et al., 2021）把声学模型与声码器合并到一个 VAE-GAN 框架，文本编码器输出的后验 z 直接送入 HiFi-GAN 风格的生成器合成波形，整个端到端可微，无需两阶段。VITS 在主观 MOS 上接近真实录音，且推理速度极快（100×+ RTF），是当前开源社区最流行的高质量 TTS 方案。后续 VITS2 / VITS3 在多说话人、情感、跨语种、长文本稳定性上继续提升。

## 四、HiFi-GAN：高效神经声码器

神经声码器从 Mel 还原波形。早期的 WaveNet 是自回归 20 层 dilated causal Conv，音质好但速度极慢（1000× 倍速比 RTF>100）。Parallel WaveGAN、WaveGlow 用 Flow 或 GAN 提升速度，但参数量和算力仍不小。

HiFi-GAN（Kong et al., 2020）是 GAN 范式声码器的代表作，设计哲学是"通过多感受野融合在大幅降低参数量同时保持音质"：

### 4.1 生成器（Generator）
上采样主体由 **3 层（或更多）Multi-Receptive Field Fusion (MRF) 残差块** 串接 4 层 转置 Conv（每层×2 上采样，合计 ×16 或 ×32，由 22.05 kHz 的 hop_size 256 / 512 决定）。每个 MRF 块并行三路不同 kernel_size / dilation_rate 的残差支路，输出相加融合，使生成器同时捕捉局部波形细节与全局包络。

### 4.2 判别器（Discriminators）
并行两类判别器：
- **Multi-Scale Discriminator (MSD)**：对原始波形、×2 降采样、×4 降采样三个尺度分别判别；
- **Multi-Period Discriminator (MPD)**：按周期 p=2,3,5,7,11 对波形折叠成 2D patch，用 2D Conv 判别不同周期结构。

### 4.3 损失函数
加权的对抗损失（LSGAN）+ 梅尔谱 L1 损失（主重建约束）+ 特征匹配损失（多层判别器隐藏层 L1）。训练 3M 参数即可达到 WaveNet 级 MOS。HiFi-GAN 的 RTF 在单张消费级 GPU 可达 500 以上，CPU 也能 20~50× 实时。目前 VITS、Bark、AudioLDM、Stable Audio 等音频生成模型几乎全部使用 HiFi-GAN 系列作为默认声码器。

## 五、音频模态未来方向

统一音频基础模型（Audio Foundation Models）正在成为新的主线：如 AudioGPT、AudioLDM 2、Stable Audio Open、Qwen-Audio、GAMA 等在一个模型内解决 ASR、TTS、语音分离、增强、音乐生成、音频描述、音频-文本跨模态检索 10+ 任务。配合 sky-v1-omni 这样的多模态统一框架，音频模态将与文本、图像、视频、3D 无缝协作，构成真正能"听、说、看、读、建"的全模态智能系统。
