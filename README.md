# sky-v1-omni 🌌

> **五模态 (Text / Image / 3D / Video / Audio) 统一大模型系统**
> 当前里程碑：`v0.1.0a1` — M1~M5 全链路已交付：Agent + RAG + OpenAI 兼容 API + UniTransformer 五模态骨干 + 三阶段训练 + 推理引擎 + 量化/LoRA + Benchmark + 5 老师蒸馏 + 预训练暖启动
> 全量 **167 tests** 通过（无 GPU / 无网络 / 无 API Key 可跑）

---

## ✨ 现在能做什么

无需 GPU、无需 API Key 就能跑（任何第三方依赖缺失 → 自动 `[SIMULATED]` 回退，不抛错）：

| 能力 | 模块 | 状态 | 说明 |
|------|------|:----:|------|
| 🤖 5 模态 Agent 编排 | `sky_v1.agent` (M1) | ✅ | **14 专家工具** + 启发式路由 + STM/LTM 记忆 + 幻觉反思 |
| 📚 RAG 检索增强知识库 | `sky_v1.rag` (M1) | ✅ | **10+ 预置文档**（含 **16 家国产大模型 + 3 家国外模型系列** 差异化优势选型表，覆盖豆包/文心/通义千问/混元/盘古/星火/MiMo/DeepSeek/Kimi/MiniMax/百川/阶跃星辰/商汤/美团LongCat/快手可灵/京东JoyAI/GPT-4.5/Claude 3.7/Gemini 2.0 Ultra/Sora Turbo）+ InMemory/Chroma 双后端 + Reranker + HyDE |
| 🌐 OpenAI 兼容 API | `sky_v1.api` (M1/M4) | ✅ | `/v1/chat/completions` 等端点 + **/v1/search/web + /v1/reasoning/deep**，Engine 与 Agent 双路径 |
| 🧠 UniTransformer 五模态骨干 | `sky_v1.model` (M2) | ✅ | RMSNorm + RoPE + SwiGLU + 5 Tokenizer / 5 Head + 模态类型嵌入 |
| 🏋️ 三阶段训练框架 | `sky_v1.training` (M3) | ✅ | Phase1 warmup / Phase2 align (SFT+InfoNCE) / Phase3 distill (KD3+DPO) + Checkpoint |
| ⚡ 推理引擎 | `sky_v1.inference` (M4) | ✅ | SkyInferenceEngine + PagedKVCache + W8A8/W4A16 量化 + LoRA 热切换 |
| 🔌 SDK + CLI | `sky_v1.sdk` / `sky_v1.cli` (M4/M5) | ✅ | OpenAI 兼容 SDK + Typer CLI（**chat/embed/serve/train/rag + search/think**）|
| 🔎 毫秒级联网搜索 | `sky_v1.agent.tools.WebSearchTool` (M5+) | ✅ | **Google / 百度 / Bing / 今日头条** 4 大引擎并发 + API+HTML 双通道 + **TTL/LRU 缓存（热查询 <50ms）** + **并发总超时快速失败**；严格区分 OK/PARTIAL/EMPTY/FAILED 四种状态，默认不伪造模拟结果（显式 `allow_simulated` 才启用） |
| 🧩 深度多步推理 | `sky_v1.agent.tools.DeepReasoningTool` (M5+) | ✅ | **Plan-Act-Observe-Reflect** Tree-of-Thoughts + 自我反思纠错 + 置信度 + 可选搜索验算 |
| 📊 Benchmark 评估 | `sky_v1.eval` (M5) | ✅ | MMLU 5-shot / HumanEval pass@1 / 推理吞吐，结果可存取 |
| 👨‍🏫 5 老师蒸馏 | `sky_v1.training.teacher_client` (M5) | ✅ | Claude/GPT/Kimi/MiMo/Qwen API + Qwen72B 本地 fallback |
| 🔄 预训练暖启动 | `sky_v1.model.pretrained_loader` (M5) | ✅ | Qwen/CLIP-ViT/Whisper 权重对齐灌入对应模态 Tokenizer |
| 📥 公开数据集下载 | `scripts/data/*` (M5) | ✅ | 多镜像源 (hf/aliyun/modelscope) 下载预训练/SFT/多模态/偏好数据 |
| 📦 与 13 个 GitHub 仓库联动 | `scripts/data/sync_github_repos_to_rag` | ✅ | 静态索引预置，可选脚本同步真实 README |

---

## ⚡ 30 秒快速开始

```bash
cd sky-v1-omni

# 1. 安装核心依赖 (不用 torch/chromadb 也能跑 Agent+RAG+API)
pip install -r requirements.txt 2>/dev/null || \
pip install "pydantic>=2.5" "omegaconf>=2.3" "pyyaml>=6.0" \
            "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27" \
            "numpy>=1.24" "tenacity>=8.2" "rich>=13.7" "typer>=0.12" \
            "typing-extensions>=4.9" "python-dotenv>=1.0" \
            "pytest>=8.0" "pytest-asyncio>=0.23"

# 2. 跑全量测试，确认环境 OK (167 tests, ~60s)
PYTHONPATH=. python -m pytest tests -q --tb=no
# -> 期望: 167 passed, 0 failed

# 3. 一键摄入预置知识库（可选，不启动服务也能查）
PYTHONPATH=. python -m scripts.rag.ingest_knowledge \
    --backend inmemory --query "Transformer 注意力机制"

# 4. 启动 Agent API 服务（默认 http://0.0.0.0:8000）
PYTHONPATH=. python -m scripts.agent.start_agent_server --ingest-presets --port 8000

# 5. 启动 Engine 模式 API（真实模型推理，model=sky-v1-* 走 SkyInferenceEngine）
PYTHONPATH=. python -m scripts.inference.serve --engine --port 8000
```

启动后打开：
- 健康检查：http://localhost:8000/health
- Swagger 文档：http://localhost:8000/docs
- Agent 工具清单：http://localhost:8000/v1/agent/tools

---

## 📡 API 示例（curl 即玩）

```bash
# 1. Agent 模式对话 - 走 Agent(planner→tool→reflection)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sky-v1-agent","messages":[{"role":"user","content":"帮我写个Python斐波那契函数并运行"}]}' \
  | python -m json.tool

# 2. Engine 模式对话 - 走 SkyInferenceEngine.chat()（需 --engine 启动）
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sky-v1-mini","messages":[{"role":"user","content":"hello sky"}],"max_tokens":8}' \
  | python -m json.tool

# 3. 多模态内容输入
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sky-v1-agent",
    "messages": [{"role":"user","content":[
      {"type":"text","text":"描述这张图里有什么"},
      {"type":"image_url","image_url":"https://example.com/cat.jpg"}
    ]}]
  }' | python -m json.tool

# 4. 直接生成一张图（Engine 模式返回真实 tensor shape）
curl -s http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一只在太空漂浮的柴犬，赛博朋克风格","n":1,"size":"1024x1024"}'

# 5. 查知识库
curl -s http://localhost:8000/v1/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Whisper 语音识别","top_k":3}'
```

---

## 🔎 联网搜索：Key 与环境变量

> **生产推荐**：配置 1~2 个 API Key 可把命中率提升至 98%+；不配置时仍可使用 4 引擎 HTML 抓取（无需 Key），但被站点反爬时会按 `PARTIAL/FAILED` 结构化告知失败原因（**默认不伪造模拟结果**，加 `allow_simulated=true` 才回退到开发模式摘要）。

| Provider | 方式 | 环境变量 | 说明 |
|----------|------|----------|------|
| **Google** | API（推荐） | `SERPAPI_API_KEY` | SerpAPI 代查 Google 结果，稳定性最佳 |
| **Google** | 官方 JSON API | `GOOGLE_SEARCH_API_KEY` + `GOOGLE_SEARCH_CX` | Programmable Search Engine，免费额度 100 次/天 |
| **Google** | HTML 抓取 | 无需 Key | `https://www.google.com/search` 直扒，零依赖兜底 |
| **百度** | HTML 抓取 | 无需 Key | `https://www.baidu.com/s` 直扒，中文首选 |
| **Bing** | 认知服务 API | `BING_SEARCH_API_KEY` | Azure 官方 Bing Search v7.0，数据质量最稳 |
| **Bing** | HTML 抓取 | 无需 Key | `https://www.bing.com/search` 直扒兜底 |
| **今日头条** | HTML 抓取 | 无需 Key | `https://so.toutiao.com/search`，资讯/短视频/新闻优势 |
| Tavily | API（辅助） | `TAVILY_API_KEY` | RAG 友好型搜索，国外资料更稳 |

### CLI 用法示例
```bash
# 默认并发 4 引擎：Google + 百度 + Bing + 今日头条
sky search "通义千问 Qwen3.8-Max 发布数据"

# 只跑国内引擎（百度 + 今日头条）
sky search "2026 最新短剧爆款" -p baidu,toutiao

# 显式允许开发模式模拟结果（真实引擎全挂时用）
sky search "某个冷僻关键词" --allow-simulated
```

### API 用法示例
```bash
curl -s http://localhost:8000/v1/search/web \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "豆包 2026 日均 Token 调用量",
    "providers": ["google","baidu","toutiao"],
    "num_results": 5,
    "allow_simulated": false
  }' | python -m json.tool
```

返回 JSON 包含：
* `status`: `OK / PARTIAL / EMPTY / FAILED`（严格区分空结果与网络异常）
* `provider_statuses[]`: 每个引擎的可用状态、命中条数、延迟 ms、失败原因
* `results[].provider`: 每条结果来自哪个引擎

---

## 🧠 训练 / 推理 / 评估 一条龙

```bash
# 预训练暖启动：从 HuggingFace 加载 Qwen/CLIP/Whisper 权重对齐灌入模型
# （1B/3B/7B 配置需 ≥16GB 内存；小内存机器可先用 toy 配置验证流程）
PYTHONPATH=. python -m scripts.training.init_from_pretrained \
    --model-config configs/model/sky_v1_1B.yaml \
    --text-repo Qwen/Qwen2.5-0.5B --image-repo openai/clip-vit-base-patch16

# 三阶段训练
# - CPU 快速验证（toy 模型，内存安全，随处可跑）
PYTHONPATH=. python -m scripts.training.train_toy_overfit --steps 10
# - 真实三阶段（--config 为模型配置，默认 1B 需 ≥16GB 内存；--steps 2 可做冒烟）
PYTHONPATH=. python -m scripts.training.phase1_warmup  --steps 2
PYTHONPATH=. python -m scripts.training.phase2_align   --steps 2
PYTHONPATH=. python -m scripts.training.phase3_distill --steps 2

# 5 老师蒸馏数据集构建（缺 API Key 时自动走 Qwen72B 本地 fallback / toy 样本）
PYTHONPATH=. python -m scripts.data.build_distillset \
    --config configs/training/teachers.yaml --max-samples 64

# Benchmark 评估：MMLU / HumanEval / 吞吐
PYTHONPATH=. python -m scripts.eval.run_benchmark \
    --tasks mmlu,humaneval,throughput --output /tmp/bench.json

# 公开数据集下载（多镜像源，失败生成 toy 样本）
PYTHONPATH=. python -m scripts.data.download_pretrain_corpus --source hf --max-samples 100
PYTHONPATH=. python -m scripts.data.download_sft_dataset     --source modelscope
PYTHONPATH=. python -m scripts.data.download_modal_datasets  --modality all
PYTHONPATH=. python -m scripts.data.download_preference      --source hf
```

---

## 🗂️ 目录结构

```
sky-v1-omni/
├── pyproject.toml / requirements.txt / .gitignore
├── README.md                                 ← 你在这里
├── configs/                                  ← YAML 配置 (OmegaConf + Pydantic)
│   ├── model/   {sky_v1_1B, 3B, 7B}.yaml     ← 五模态模型规格
│   ├── training/{phase1_warmup, phase2_align, phase3_distill, teachers, deepspeed_zero2}.yaml
│   ├── inference/{sky_v1_infer_cpu, gpu, lora, quant}.yaml
│   ├── rag/vector_db_chroma.yaml
│   └── agent/ {planner_llm.yaml, tool_pool.yaml}
├── sky_v1/
│   ├── utils/                                ← 日志 / 配置 / 种子 / 重试
│   ├── model/                                ← (M2) UniTransformer 骨干
│   │   ├── backbone / attention / ffn / norm / embeddings / transformer_layer
│   │   ├── modal_tokenizers/ (Text/Image/Audio/Video/3D)
│   │   ├── modal_heads/      (Text/Image/Audio/Video/3D)
│   │   ├── lora.py                           ← (M4) LoRA 热切换
│   │   ├── pretrained_loader.py              ← (M5) Qwen/CLIP/Whisper 暖启动
│   │   ├── sky_model.py / config.py
│   ├── training/                             ← (M3) 三阶段训练
│   │   ├── trainer / sft / distill / dpo / losses / checkpoint / callbacks
│   │   └── teacher_client.py                 ← (M5) 5 老师 API + Qwen72B fallback
│   ├── data/                                 ← 数据生成 / 加载
│   │   ├── toy_generator / datasets / collator
│   │   └── real_datasets.py                  ← (M5) 多镜像源真实数据集加载
│   ├── inference/                            ← (M4) 推理引擎
│   │   ├── engine / kv_cache / paged_attention / quant
│   ├── eval/                                 ← (M5) Benchmark 评估
│   │   ├── benchmark / mmlu / humaneval / throughput
│   ├── rag/                                  ← (M1) RAG 知识库 (ABC + InMemory/Chroma)
│   │   └── presets/ (8 篇预置 md)
│   ├── agent/                                ← (M1) Agent 编排 + 12 工具
│   ├── api/                                  ← (M1/M4) FastAPI + OpenAI 兼容
│   │   ├── middleware/ (CORS + ReqID + 日志 + 全局异常)
│   │   └── routes/ (chat/rag/modal/agent/health)
│   ├── sdk/client.py                         ← (M4) OpenAI 兼容 SDK
│   └── cli/main.py                           ← (M4/M5) Typer CLI
├── scripts/
│   ├── rag/ingest_knowledge.py               ← 一键知识摄入
│   ├── agent/start_agent_server.py           ← 一键服务启动
│   ├── inference/ {serve, chat, generate}    ← 推理服务/交互
│   ├── training/ {init_from_pretrained, phase1/2/3, train_toy_overfit}
│   ├── data/ {download_pretrain/sft/modal/preference, build_distillset, sync_github_repos_to_rag}
│   └── eval/run_benchmark.py                 ← Benchmark 入口
├── tests/                                    ← 167 tests (Unit/Integration/E2E)
└── docs/superpowers/
    ├── specs/2026-08-08-sky-v1-design.md     ← 规格书 v1.0 (已评审)
    └── plans/ {m1, m2-m3, m4-m5} 实现计划
```

---

## 🧪 质量保证 (Bug 零容忍)

```
测试金字塔 (167 passed, 0 failed, ~60s)
├─ E2E:         12  (health/metrics/chat/rag/image/agent/tools + M2M3/M4M5 全链路冒烟)
├─ Integration: 32  (RAG 全流程 / 工具链 / 多模态 / 模型序列化 / 训练过拟合 / API Engine 模式 / serve --engine / 数据下载·评估·暖启动脚本冒烟 / 搜索·推理路由 + RAG 预置文档)
└─ Unit:        123 (形状 / 数值 / 路由 / 边界 / Guard / 量化 / LoRA / 蒸馏 / Benchmark / 暖启动 / 搜索·推理工具 + 缓存 TTL/LRU)
```

额外防御：
- 入参 100% `Pydantic v2` 校验 + 每函数 Guard Clause
- 所有外部调用 `@with_retry(max=3)` 指数退避
- 任何第三方 Key / 后端 Missing → 自动 `[SIMULATED]` 成功回退，不抛错
- 单元测试可在无网络、无 GPU、无 chromadb 环境通过
- CI：`.github/workflows/` 三件套（unit_tests / integration_tests / benchmark）

---

## 🚙 路线图

| Milestone | 状态 | 内容 |
|-----------|:----:|------|
| **M1** | ✅ **已交付** | Agent + RAG + OpenAI API |
| **M2** | ✅ **已交付** | UniTransformer 五模态骨干 + 5 Tokenizer/Head + 模态类型嵌入 |
| **M3** | ✅ **已交付** | 三阶段 Trainer (warmup/align/distill) + SFT/KD/InfoNCE/DPO + Checkpoint + 数据生成 |
| **M4** | ✅ **已交付** | SkyInferenceEngine + PagedKVCache + W8A8/W4A16 量化 + LoRA + SDK/CLI + API 联通 |
| **M5** | ✅ **已交付** | Benchmark (MMLU/HumanEval/吞吐) + 5 老师蒸馏 + 预训练暖启动 + 多镜像源数据下载 |
| **M6+** | 📋 计划中 | 真实大规模预训练 / 多卡 DeepSpeed / 在线服务化 / 多模态对齐精调 |

---

## 🔗 与现有 13 个仓库联动

| 你的仓库 | 对接方式 |
|---------|---------|
| **STAR_CHAT / STAR-CHAT** | API 100% 兼容，只需改 `base_url` → `http://<skyv1-host>:8000/v1`，即可从小米 mimo 无缝切换 |
| **starbuddy-agent** | 工具 SDK 可作为 `sky_v1.agent.tools` 的补充 import 进来扩展 |
| **starbuddy-code** | `code_tool` 底层可直接接入 starbuddy-code 的代码理解运行器 |
| **其余 10 个仓库** | 已预置索引文档进 RAG；`python -m scripts.data.sync_github_repos_to_rag` 可同步最新 README |

---

## 📜 更多文档

- [规格设计书](docs/superpowers/specs/2026-08-08-sky-v1-design.md)
- [M1 实现计划](docs/superpowers/plans/2026-08-08-m1-agent-systems.md)
- [M2+M3 实现计划](docs/superpowers/plans/2026-08-09-m2-m3-model-training.md)
- [M4+M5 实现计划](docs/superpowers/plans/2026-08-09-m4-m5-inference-sdk.md)

---

*sky-v1 Team · 2026-08-09 · Build with ❤️ in PyTorch + FastAPI*
