# Sky V1 Omni Desktop

基于 [electron-vite](https://evite.netlify.app/) 构建的跨平台桌面客户端，封装了 `sky-v1-omni` Python 后端，提供 VSCode 风格的 Chat / 搜索 / 深度推理 / RAG / GitHub / 终端一体化工作台。

## 版本锚点（重要）

这些版本组合已验证可协同工作：

| 包 | 版本区间 |
|---|---|
| electron-vite | `^2` |
| Vite | `^5` |
| @vitejs/plugin-react | `^4` |
| TypeScript | `^5.4` |
| Electron | `^30` |
| electron-builder | `^24` |
| pnpm | 建议 8/9（使用 `.npmrc` 的 hoisted 模式） |

## 前置条件

1. **Node.js ≥ 18** + **pnpm**
2. **Python ≥ 3.10**
   - 确保 `sky-v1-omni` 仓库已安装依赖（建议在仓库根目录创建 venv 并激活）
   - `python -m sky_v1.cli.main serve` 必须可以被执行（主进程会自动把仓库根加入 `PYTHONPATH`）
3. **GitHub CLI (gh)**（可选，用于 GitHub Tab）：
   - `gh auth login`
   - 若不登录 GitHub 标签页会提示 Not logged in，但其它功能不受影响
4. **node-pty 编译工具链**（native 模块）：
   - **Windows**：Visual Studio Build Tools (C++ workload) + Python
   - **macOS**：Xcode Command Line Tools (`xcode-select --install`)
   - **Linux**：build-essential / python3 / make / gcc

## 安装依赖

桌面项目是**自包含**的，不是 pnpm workspace root：

```bash
cd desktop
pnpm install
```

`.npmrc` 中已启用：
```
shamefully-hoist=true
node-linker=hoisted
```
这能避免 Electron 原生模块（node-pty 等）出现的 `.pnpm` phantom dependency / dll 搜索路径问题。

## 开发模式

```bash
cd desktop
pnpm dev
```

- 自动启动：Main / Preload 热更新 + Renderer Vite HMR
- Vite dev server 通过 `ELECTRON_RENDERER_URL` 注入
- 主进程会自动在 `127.0.0.1:8765` 起（或占用时自动递增）后端 Python 服务

## 构建 & 打包

```bash
cd desktop
pnpm build        # 输出到 dist/（main/preload/renderer 三个子目录）
pnpm preview      # 用打包后的代码启动 electron 预览（不安装）
pnpm pack         # electron-builder --dir：生成未打包的可执行目录
pnpm dist         # 生产出安装包：dmg / nsis / AppImage 到 release/
```

产物目录：
```
desktop/
  dist/             # electron-vite 构建结果
  release/          # electron-builder 打包产物（dmg、exe、AppImage 等）
```

## 图标 & 构建资源（占位）

`desktop/build/` 为 electron-builder 资源目录。放置以下文件以替换默认图标：

```
desktop/build/
  icon.icns        # macOS (512x512 @2x)
  icon.ico         # Windows (256x256)
  icon.png         # Linux (512x512)
```

> 当前仓库不含图标文件，`electron.vite.config.ts` 中 `icon` 路径会在缺失时由 Electron 使用默认主题图标兜底。

## 应用结构

```
desktop/
├─ electron.vite.config.ts      # 3 bundle 配置：main / preload / renderer
├─ electron-builder.yml         # 多平台打包 + GitHub publish
├─ tsconfig.json / tsconfig.node.json
├─ src/
│  ├─ main/index.ts             # Main: BrowserWindow、后端子进程、IPC、pty、autoUpdater
│  ├─ preload/index.ts          # contextBridge exposeInMainWorld('sky', ...)
│  └─ renderer/
│     ├─ index.html
│     └─ src/
│        ├─ main.tsx            # React 入口
│        ├─ App.tsx             # VSCode-style 布局：Activity Bar · Side · Tabs · Panel · Status
│        ├─ store.ts            # React Context + useReducer 全局 store（tabs/activity/backend...）
│        ├─ vite-env.d.ts       # window.sky TS 类型声明
│        └─ views/
│           ├─ ChatView.tsx         # /v1/chat/completions
│           ├─ SearchView.tsx       # /v1/search/web（多 provider、top N）
│           ├─ ReasoningView.tsx    # /v1/reasoning/deep（Plan → Steps → Answer 折叠式）
│           ├─ RagView.tsx          # rag:list-files / ingest / /v1/rag/query
│           ├─ GithubView.tsx       # gh cli 代理（auth / repos / issues）
│           ├─ TerminalView.tsx     # xterm.js + node-pty
│           └─ SettingsView.tsx     # 版本、后端控制、日志
```

## 安全沙箱 & IPC 约定

- `contextIsolation: true`, `sandbox: true`, `nodeIntegration: false`
- 所有渲染进程调用经 `window.sky.*`（contextBridge），通过 IPC 到主进程
- 后端 HTTP 代理统一经 `api:fetch` 主通道，避免 `127.0.0.1` 在 sandbox 下混合内容问题
- GitHub 命令由主进程 `child_process.exec('gh ...')` 执行并返回 stdout/stderr/code

## 常见问题

1. **后端启动失败 / port 被占用**
   - 主进程会自动探测可用端口（8765→+1），状态栏会显示实际监听端口
   - 可在 Settings 页面看 `Recent Backend Logs` 定位 Python 模块导入问题

2. **node-pty 编译失败**
   - Windows：务必打开 Developer Command Prompt 或安装 VS Build Tools C++ workload
   - macOS：`xcode-select --install`，并确保 `pnpm rebuild node-pty` 能通过

3. **pnpm install 很慢或出现 native 找不到**
   - 不要从 workspace root 给 desktop 装依赖。desktop 是独立目录，必须 `cd desktop && pnpm i`
   - 删除 `desktop/node_modules` 和 `desktop/pnpm-lock.yaml` 后重跑

4. **打包体积过大**
   - `electron-builder.yml` 中 `files: ["dist/**/*", "package.json"]` 已只打包必要产物
   - 若要更小，可额外启用 `compression: maximum` 或做 asar 拆包
