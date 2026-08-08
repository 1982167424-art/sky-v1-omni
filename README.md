# sky-v1-omni 🌌

> **五模态 (Text / Image / 3D / Video / Audio) 大模型系统**  
> 当前里程碑：`v0.1.0a1 (M1)` - Backup Agent + RAG 知识库 + OpenAI 兼容 API 上线可用  
> 训练框架 (UniTransformer + 三阶段蒸馏) - 后续里程碑

---

## ✨ M1 现在能做什么

无需 GPU、无需 API Key 就能跑：

| 能力 | 实现状态 | 说明 |
|------|:--------:|------|
| 🤖 5 模态 Agent 编排 | ✅ 可用 | 12 个专家工具（3Text + 2Image + 2Audio + 2Video + 3×3D），启发式路由，零网络依赖 |
| 📚 RAG 检索增强知识库 | ✅ 可用 | 8 篇预置技术文档 (67 Chunks) + InMemory / Chroma 双后端 + Reranker |
| 🌐 OpenAI 兼容 API | ✅ 可用 | `/v1/chat/completions` 等 10+ 端点，STAR_CHAT 改 base_url 即可接入 |
| 🧠 长期 + 短期记忆 | ✅ 可用 | 滚动窗口 STM（20 turns）+ 向量库 LTM |
| 🪞 自我反思（幻觉检测） | ✅ 可用 | 硬断言 / 过短 / 关键词模式捕获 rewrite 触发 |
| 📦 与 13 个 GitHub 仓库联动 | ✅ 可用 | 静态索引已预置，可选脚本同步真实 README |
| 🔧 CLI 三件套 | ✅ 可用 | 知识摄入 / 服务启动 / GitHub 同步 |

---

## ⚡ 30 秒快速开始

```bash
cd sky-v1-omni

# 1. 安装核心依赖 (不用chromadb也能跑)
pip install -r requirements.txt 2>/dev/null || \
pip install "pydantic>=2.5" "omegaconf>=2.3" "pyyaml>=6.0" \
            "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27" \
            "numpy>=1.24" "tenacity>=8.2" "rich>=13.7" \
            "typing-extensions>=4.9" "python-dotenv>=1.0" \
            "pytest>=8.0" "pytest-asyncio>=0.23"

# 2. 先跑一次测试，确认环境 OK
PYTHONPATH=. python -m pytest tests -x -q 2>&1 | tail -5
# -> 期望: 59 passed, 0 failed

# 3. 一键摄入预置知识库（可选，不启动服务也能查）
PYTHONPATH=. python -m scripts.rag.ingest_knowledge \
    --backend inmemory --query "Transformer 注意力机制"

# 4. 启动 Agent API 服务（默认 http://0.0.0.0:8000）
PYTHONPATH=. python -m scripts.agent.start_agent_server --ingest-presets --port 8000
```

启动后打开：
- 健康检查：http://localhost:8000/health
- Swagger 文档：http://localhost:8000/docs
- OpenAPI Schema：http://localhost:8000/openapi.json
- Agent 工具清单：http://localhost:8000/v1/agent/tools

---

## 📡 API 示例（curl 即玩）

```bash
# 1. 通用对话 - 走 Agent(planner→tool→reflection)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sky-v1-agent","messages":[{"role":"user","content":"帮我写个Python斐波那契函数并运行"}]}' \
  | python -m json.tool

# 2. 多模态内容输入
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sky-v1-agent",
    "messages": [{"role":"user","content":[
      {"type":"text","text":"描述这张图里有什么"},
      {"type":"image_url","image_url":"https://example.com/cat.jpg"}
    ]}]
  }' | python -m json.tool

# 3. 直接生成一张图（模态端点）
curl -s http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一只在太空漂浮的柴犬，赛博朋克风格","n":1,"size":"1024x1024"}'

# 4. 查知识库
curl -s http://localhost:8000/v1/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Whisper 语音识别","top_k":3}'

# 5. 执行 Agent 单步
curl -s -X POST http://localhost:8000/v1/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"user_message":"把“你好，世界”转语音","session_id":"demo001"}'
```

---

## 🗂️ 目录结构 (M1 已交付部分)

```
sky-v1-omni/
├── pyproject.toml / requirements.txt / .gitignore
├── README.md                                 ← 你在这里
├── configs/                                  ← YAML 配置 (OmegaConf + Pydantic)
│   ├── rag/vector_db_chroma.yaml
│   └── agent/ {planner_llm.yaml, tool_pool.yaml}
├── sky_v1/
│   ├── utils/                                ← 日志 / 配置 / 种子 / 重试
│   ├── rag/                                  ← RAG 知识库 (ABC + InMemory/Chroma)
│   │   └── presets/ (8 篇预置 md, 共~13000字)
│   ├── agent/                                ← Agent 编排 + 12 工具
│   │   └── tools/ (Text / Image / Audio / Video / 3D)
│   └── api/                                  ← FastAPI + OpenAI 兼容类型
│       ├── middleware/ (CORS + ReqID + 日志 + 全局异常)
│       └── routes/ (chat/rag/modal/agent/health)
├── scripts/
│   ├── rag/ingest_knowledge.py               ← 一键知识摄入
│   ├── agent/start_agent_server.py           ← 一键服务启动
│   └── data/sync_github_repos_to_rag.py      ← 同步 13 仓库到 KB
├── tests/                                    ← 59 tests (Unit/Integration/E2E)
└── docs/superpowers/
    ├── specs/2026-08-08-sky-v1-design.md     ← 规格书 v1.0 (已评审)
    └── plans/2026-08-08-m1-agent-systems.md  ← M1 实现计划
```

---

## 🧪 质量保证 (Bug 零容忍)

```
测试金字塔 (59 passed)
├─ E2E: 7  (每次启动必跑, <5 min)  ← health / metrics / chat / rag / image / agent / tools
├─ Integration: 6  (nightly, <30 min)  ← RAG 全流程 / 工具链 / 多模态
└─ Unit: 45  (每 PR, <5 min)  ← 形状 / 数值 / 路由 / 边界 / Guard
```

额外防御：
- 入参 100% `Pydantic v2` 校验 + 每函数 Guard Clause
- 所有外部调用 `@with_retry(max=3)` 指数退避
- 任何第三方 Key / 后端 Missing → 自动 `[SIMULATED]` 成功回退，不抛错
- 单元测试可在无网络、无 GPU、无 chromadb 环境通过

---

## 🚙 路线图

| Milestone | 状态 | 内容 |
|-----------|:----:|------|
| **M1** | ✅ **已交付** | Agent + RAG + OpenAI API + 59 Tests |
| **M2** | 📋 计划中 | UniTransformer 架构 + 5 Tokenizer/Head + 三阶段 Trainer 代码 + 2-step 过拟合测试 |
| **M3** | 📋 计划中 | 所有脚本/CI/预置 SDK/文档 全就位 |
| **M4** | 📋 计划中 | 推理服务 + API 联通 + 训练/Agent 双路径混合模式 E2E 冒烟 |

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

---

*sky-v1 Team · 2026-08-08 · Build with ❤️ in PyTorch + FastAPI*
