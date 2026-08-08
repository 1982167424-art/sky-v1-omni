# sky-v1-omni 生态 13 个 GitHub 仓库索引简介

本索引文档梳理 sky-v1 团队在 GitHub 上维护/协作的 13 个核心仓库，用于 RAG 知识库的代码检索入口与 Agent 工具链的上下文。

## 总览表

| # | 仓库名 | 主语言 | 仓库定位 | 与 sky-v1-omni 的关系 |
|---|--------|--------|----------|------------------------|
| 1 | sky-v1-omni | Python (核心) | 5-modal Omni Model 主仓库（文本/图像/3D/视频/音频） | **当前仓库：主框架本体** |
| 2 | IKNOW | Python | 数据集构建、知识增强与评测工具包 | 上游数据生产器 → 喂给 omni 的 SFT/DPO 数据 |
| 3 | pppdlw | Python | 大规模并行训练的轻量封装（pp + deepspeed + lightning w） | omni 预训练/微调训练器底层 |
| 4 | ppdbeta | JavaScript | pppdlw 的 Web 控制台，可视化 loss、激活、样本 | omni 训练看板与调参前端 |
| 5 | SKYOS | JavaScript | skyOS 下一代操作系统 Web Shell（SKYOS V1） | omni 模型作为 Agent Kernel 运行的宿主 OS |
| 6 | SKY-OS | 混合（Shell + 多语言） | skyOS V0 原型，Shell scripts + 桌面 UI 概念验证 | SKYOS 上游原型代码参考 |
| 7 | 1982167424-art | 纯 Profile 仓库（README / Gallery） | sky-v1 团队核心成员 1982167424 的个人主页、作品集、模型演示 | 团队成员档案，含模型生成素材 Gallery |
| 8 | STAR_CHAT | TypeScript | AI 聊天助手前端（基于小米 mimo SDK + 多模型路由） | omni API 的首个集成前端 |
| 9 | STAR-CHAT | TypeScript | STAR_CHAT 主仓库的镜像/二次开发分支（历史兼容仓库） | 旧版兼容代码，保留给老用户 |
| 10 | tanshi-app | TypeScript | 叹思 App：轻量移动端 AI 应用，1 星起步阶段 | omni 移动版推理与 TTS/ASR 入口 |
| 11 | starbuddy-agent | Python + MCP | StarBuddy Agent Tooling：MCP Skills 与 Agent 工作流编排 | omni Agent API 的标准工具库 |
| 12 | starbuddy-code | Python + Tree-sitter | StarBuddy Code Tooling：代码库索引、AST 检索、Diff 生成 | omni Code Interpreter 与 Dev Agent 后端 |
| 13 | ai-introduction | HTML | 面向中学生的 AI 科普网站，含 1 个 issue | omni 公共关系 / 教育侧的落地展示 |

---

## 仓库逐一说

### 1. sky-v1-omni（当前仓库）
- **定位**：sky-v1 0.1.0-M1 里程碑核心仓库。实现 5-modal（Text/Image/3D/Video/Audio）统一大模型的训练脚手架、RAG 知识库子系统（`sky_v1/rag/`，正是您在读的代码！）、OpenAI 兼容 API Server（`sky_v1/api/`）、Agent Loop（`sky_v1/agent/`）、CLI 工具（`scripts/`）。
- **关键模块**：`sky_v1.utils`（日志、配置、种子、重试）、`sky_v1.rag`（本 RAG 子系统）、`sky_v1.models`（统一骨干）、`sky_v1.training`（三阶段训练）、`sky_v1.api`（FastAPI 服务）、`sky_v1.agent`（Agent + MCP Skill）。
- **入口脚本**：`skyv1-rag-ingest` / `skyv1-agent-server` / `skyv1-sync-github-repos-to-rag`。
- **与其他仓库协同**：本仓库的训练层依赖 pppdlw；数据层依赖 IKNOW；Agent 工具层依赖 starbuddy-agent / starbuddy-code；前端展示层由 STAR_CHAT / ppdbeta / SKYOS 对接。

### 2. IKNOW
- **定位**：IKNOW = "Intelligent Knowledge for NLP mOdels Working"。数据侧平台仓库，负责从网页、GitHub、论文、电子书、多模态数据源爬取、清洗、去重、合成指令对、构造偏好数据集。
- **主要管线**：
  - Common Crawl → 语言分类 → 去重（MinHash LSH）→ 质量打分器（基于 GPT-4 二分类器的蒸馏版轻量模型）→ 困惑度过滤 → 最终预训练语料；
  - Self-Instruct + Evol-Instruct 合成 SFT 数据；
  - 基于 ORCA / UltraFeedback 的 5 老师偏好蒸馏 → DPO 数据；
  - 多模态 captioning pipeline（BLIP-2 / ShareGPT4V / OCR）→ 图/视频/音频字幕。
- **与 omni 的关系**：IKNOW 产物 `iknow_release_v1.parquet` 是 omni 预训练、SFT、DPO 三个阶段的直接输入。

### 3. pppdlw
- **定位**：pppdlw = "PyTorch + PP (Pipeline Parallel) + DeepSpeed + Lightning Wrapper"。大规模分布式训练框架的轻量封装。
- **核心能力**：ZeRO-1/2/3 配置化；张量并行（Megatron-LM）集成；流水线并行 GPipe 调度；混合精度 BF16/FP8；FlashAttention v2/v3 自动切换；Checkpoint 断点续训；wandb / tensorboard 双写；离线离线评估 hook。
- **与 omni 的关系**：`sky_v1.training` 模块直接 import `from pppdlw import Trainer`，训练配置以 yaml 形式放在 `sky_v1/conf/training/*.yaml`。

### 4. ppdbeta（JavaScript）
- **定位**：pppdlw 的 Web Dashboard（Beta 版）。Node.js + Vue 3 实现，WebSocket 实时推送训练指标。
- **功能**：Loss 曲线、学习率曲线、梯度范数、激活分布直方图、每 N 步随机采样生成样例、Bad Case 可视化（图文/音频/视频嵌入）、节点资源热力图（GPU 利用率/显存/网络）、超参对比面板（支持最多 8 次跑批 overlay 对比）。
- **与 omni 的关系**：每次 omni 训练启动会往 ppdbeta 发一个注册事件，训练过程中每 10 步推一次指标，训练结束后产出报告链接同步写入 Notion / Lark Wiki。

### 5. SKYOS（JavaScript）
- **定位**：SKYOS V1 = "sky-v1 Operating System"。基于浏览器的下一代"AI 原生操作系统"Web Shell，把应用变成"智能体可组合的 Skill 包"。
- **核心特性**：
  - 类 Nushell 的命令行，每个命令是一个 MCP Tool，可以组合管道 `find ~/docs | rag-query "Transformer 注意力" | llm-summarize --style=bullet | open-slide`；
  - 多窗口 Web Desktop，每个 App 渲染一个 IFrame，跨 App 拖拽由 Agent Kernel 做语义级解释；
  - Agent Kernel 通过 WebSocket 连接 omni 的 Agent API Server。
- **与 omni 的关系**：SKYOS 是 omni 的"壳"，omni 是 SKYOS 的"脑"。

### 6. SKY-OS（混合）
- **定位**：SKYOS V0 原型仓库。用 Bash + 简单的 Electron Shell + Linux Namespace 技术做的早期 PoC，验证"AI 原生命令行 + 容器化隔离"概念。
- **状态**：进入只读归档阶段，不再接受新功能 PR；仅在需要参考历史设计决策（如 Ns 权限设计、FS 挂载钩子）时查阅。
- **与 omni 的关系**：SKYOS V1 的部分 shell 语法（管道语义、Skill 注册格式）沿用自 SKY-OS V0。

### 7. 1982167424-art（Profile 仓库）
- **定位**：sky-v1 核心贡献者 `@1982167424` 的个人作品集仓库。纯静态 GitHub Pages + README Gallery。
- **内容**：该成员产出的图像生成样例（Stable Diffusion / FLUX 调参实验）、3D 作品（Blender + 3DGS 重建）、视频生成样例（Sora/CogVideo）、模型蒸馏对比表格、技术博客译文。
- **与 omni 的关系**：收录到 RAG 中作为"高质量视觉案例检索来源"，Agent 在生成 prompt 模板、调参建议时可引用这些被人工验证过的优质样例。

### 8. STAR_CHAT（TypeScript）
- **定位**：StarChat = "sky-v1 Team's AI Chat"。基于小米 mimo 前端 SDK（React + TypeScript）+ 自研多模型路由后端的 AI 聊天产品。
- **特性**：多模型切换（omni 主模型、Qwen、Llama 3 作为兜底）、System Prompt 预设商城（100+ 角色预设）、多模态上传（图片/音频/文档）→ RAG 检索、对话导出 Markdown、协同编辑（多人共享同一会话链接）。
- **与 omni 的关系**：首个集成 sky-v1-omni Chat Completions API 的生产级前端；还负责 AB 测试 omni 的不同 SFT/DPO checkpoint 在真实用户对话上的胜率。

### 9. STAR-CHAT（TypeScript，第二个同名仓库）
- **定位**：历史镜像仓库。由于组织迁移（从个人账户 org 搬到 sky-v1 org）时 fork 关系丢失而保留的兼容只读镜像。
- **内容**：复刻 V0 版本的代码提交历史，issue/PR 关闭状态与主仓一致。
- **与 omni 的关系**：仅保证老链接不 404，不在本仓合代码。

### 10. tanshi-app（TypeScript）
- **定位**：叹思（TanSi）轻量移动端 AI 应用。Capacitor + Ionic + TypeScript，一套代码打包 iOS / Android / 小程序。
- **当前状态**：立项早期（1 星阶段），已打通"omni API 流式对话 + Whisper cpp 端上 ASR + 端上 HiFi-GAN Lite TTS"最小闭环。
- **与 omni 的关系**：作为 omni 在移动端的低带宽 / 离线模式入口，端上仅跑 ASR/TTS，语言模型与 RAG 仍走云端 omni API。

### 11. starbuddy-agent（Python + MCP）
- **定位**：StarBuddy = sky-v1 的 Agent 工具 SDK。严格实现 MCP（Model Context Protocol）Server/Client 标准，内置 50+ 通用 Agent Skill。
- **内置工具类**：
  - 通用：文件读写、Shell 执行（带沙箱白名单）、Web 抓取（Jina Reader API）、代码沙箱（Pyodide）；
  - 数据：Google Sheets / Lark Base / Notion Database 读写；
  - 日程：Google Calendar / Lark Calendar / Outlook 日程创建与查询；
  - 消息：Lark / 企业微信 / Slack / Telegram 消息发送；
  - RAG：sky_v1.rag 子系统的 MCP 包装（增删查知识库）。
- **与 omni 的关系**：`sky_v1.agent` 模块的 `SkillManager` 直接 import `starbuddy_agent` 注册全部工具。

### 12. starbuddy-code（Python + Tree-sitter）
- **定位**：StarBuddy Code = 面向软件开发 Agent 的代码理解工具集。
- **核心能力**：
  - 基于 Tree-sitter 的多语言（Python/TS/Go/Rust/Java/C++）AST 索引，支持"查找符号定义/引用/调用图"；
  - 仓库级 Semantic Search：用 bge-m3 对每个函数、类、方法做离线嵌入，支持自然语言查代码；
  - PR 级 Diff Review Agent：结合 Brooks Lint（12 本经典工程书规则）输出结构化 Review 评论；
  - Commit Message 生成、Git Worktree 多分支并行 Agent 执行。
- **与 omni 的关系**：本仓库（sky-v1-omni）的 CI Bot 就用 starbuddy-code 跑每 PR 的自动化 Review；同时作为 omni Code Interpreter 模式的底层。

### 13. ai-introduction（HTML）
- **定位**：面向中学生的 AI 科普网站。纯静态 HTML + CSS + 一点点 JS，无构建工具。
- **内容**：什么是 AI、什么是机器学习、什么是神经网络、大语言模型原理（漫画图解）、Stable Diffusion 动手小实验、AI 伦理、sky-v1 团队介绍。
- **issue 状态**：1 个未关闭 issue（2024 级高一学生提的），请求增加"如何用 Python 训练一个手写数字识别 CNN"的章节。
- **与 omni 的关系**：作为 omni 公共教育侧的落地展示；RAG 收录该网站全文，Agent 在被问到"给我孩子解释 AI 是什么"时优先检索这里的语言风格与素材。

---

## 13 个仓库的依赖关系图（文字版）

```
[ai-introduction]            [1982167424-art]        [tanshi-app]
       |                            |                     |
       ▼                            ▼                     ▼
[STAR_CHAT / STAR-CHAT镜像]  ←  [sky-v1-omni]  →  [SKYOS / SKY-OS(V0)]
                                  ↑    ↑                    ↑
                         IKNOW ───┘    └── pppdlw          │
                                              ↑             │
                                        ppdbeta (看板)      │
                                                            │
                             starbuddy-agent ───────────────┘
                             starbuddy-code  ───────────────┘
```

Agent 在定位"我需要改哪里"时，可按本索引中的"与 omni 的关系"列快速路由：改 RAG 能力 → sky-v1-omni/sky_v1/rag；改训练 → pppdlw + sky-v1-omni/sky_v1/training；改数据 → IKNOW；改前端聊天 → STAR_CHAT；改 Agent 工具 → starbuddy-agent；改代码评审 → starbuddy-code；改操作系统宿主 → SKYOS。
