import { app, BrowserWindow, ipcMain, shell } from 'electron'
import { spawn, execFile, ChildProcess } from 'child_process'
import { createServer } from 'net'
import * as path from 'path'
import * as fs from 'fs'
import { autoUpdater } from 'electron-updater'
import { IPty, spawn as ptySpawn } from 'node-pty'

let mainWindow: BrowserWindow | null = null
let backendProcess: ChildProcess | null = null
let backendPort: number = 8765
let backendRunning: boolean = false
let backendLogs: string[] = []
const MAX_LOG_LINES = 500

let ptyProcess: IPty | null = null

const repoRoot = path.resolve(__dirname, '..', '..', '..')

// ===========================================================================
// Multi-Provider LLM API Key 管理 & 流式 Chat 客户端
// 支持：火山方舟(Ark) · 小米(MiLM) · 阿里通义(DashScope) · 智谱(GLM) · 月之暗面(Moonshot) · DeepSeek · OpenAI 兼容
// ===========================================================================

export interface ProviderMeta {
  id: string
  name: string
  brandColor: string
  defaultBaseUrl: string
  docsUrl: string
  // 火山的 model 实际是「推理接入点 ID (endpoint_id)」，UI 上要明确提示
  modelPlaceholder: string
  modelLabel: string
  // 是否官方支持 SSE stream
  supportsStreaming: boolean
  defaultModel?: string
}

export const PROVIDER_META: ProviderMeta[] = [
  {
    id: 'volcengine',
    name: '火山引擎方舟 (Ark)',
    brandColor: '#1664FF',
    defaultBaseUrl: 'https://ark.cn-beijing.volces.com/api/v3',
    docsUrl: 'https://www.volcengine.com/docs/82379',
    modelLabel: '推理接入点 Endpoint ID',
    modelPlaceholder: 'ep-2024xxxxxxxxxxxxxxxx (例如 ep-20250801123456-abcde)',
    supportsStreaming: true,
    defaultModel: ''
  },
  {
    id: 'minimax',
    name: 'MiniMax (M3)',
    brandColor: '#FF6900',
    defaultBaseUrl: 'https://api.minimax.io/v1',
    docsUrl: 'https://platform.minimaxi.com/docs',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 MiniMax-M3',
    supportsStreaming: true,
    defaultModel: 'MiniMax-M3'
  },
  {
    id: 'dashscope',
    name: '阿里云 通义千问 (DashScope)',
    brandColor: '#FF6A00',
    defaultBaseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    docsUrl: 'https://help.aliyun.com/zh/dashscope',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 qwen-plus / qwen-turbo / qwen-max',
    supportsStreaming: true,
    defaultModel: 'qwen-plus'
  },
  {
    id: 'zhipu',
    name: '智谱 AI (GLM / 清言)',
    brandColor: '#0080FF',
    defaultBaseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    docsUrl: 'https://open.bigmodel.cn/dev/api',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 glm-4-plus / glm-4-air',
    supportsStreaming: true,
    defaultModel: 'glm-4-plus'
  },
  {
    id: 'moonshot',
    name: '月之暗面 Moonshot (Kimi)',
    brandColor: '#3370FF',
    defaultBaseUrl: 'https://api.moonshot.cn/v1',
    docsUrl: 'https://platform.moonshot.cn/docs',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 moonshot-v1-8k / moonshot-v1-128k',
    supportsStreaming: true,
    defaultModel: 'moonshot-v1-8k'
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    brandColor: '#4143B3',
    defaultBaseUrl: 'https://api.deepseek.com/v1',
    docsUrl: 'https://api-docs.deepseek.com',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 deepseek-chat / deepseek-reasoner',
    supportsStreaming: true,
    defaultModel: 'deepseek-chat'
  },
  {
    id: 'nvidia',
    name: 'NVIDIA NIM',
    brandColor: '#76b900',
    defaultBaseUrl: 'https://integrate.api.nvidia.com/v1',
    docsUrl: 'https://docs.api.nvidia.com',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 moonshotai/kimi-k2.6 / deepseek-ai/deepseek-r1',
    supportsStreaming: true,
    defaultModel: 'moonshotai/kimi-k2.6'
  },
  {
    id: 'openai-compatible',
    name: 'OpenAI 兼容（自定义 Base URL）',
    brandColor: '#10a37f',
    defaultBaseUrl: 'https://api.openai.com/v1',
    docsUrl: 'https://platform.openai.com/docs',
    modelLabel: '模型 / 部署名',
    modelPlaceholder: '例如 gpt-4o-mini / 或你自部署的模型名',
    supportsStreaming: true,
    defaultModel: 'gpt-4o-mini'
  }
]

export interface ProviderConfig {
  id: string
  apiKey: string
  baseUrl: string
  model: string
  enabled: boolean
  temperature?: number
  maxTokens?: number
}

export interface ProvidersStoreShape {
  version: 1
  activeProvider: string
  providers: Record<string, ProviderConfig>
}

function providersStorePath(): string {
  const userData = app.getPath('userData')
  if (!fs.existsSync(userData)) fs.mkdirSync(userData, { recursive: true })
  return path.join(userData, 'providers.json')
}

function maskSecret(v: string): string {
  if (!v) return ''
  if (v.length <= 8) return v.slice(0, 2) + '****'
  return v.slice(0, 4) + '****' + v.slice(-4)
}

function defaultStore(): ProvidersStoreShape {
  const providers: Record<string, ProviderConfig> = {}
  for (const m of PROVIDER_META) {
    providers[m.id] = {
      id: m.id,
      apiKey: '',
      baseUrl: m.defaultBaseUrl,
      model: m.defaultModel || '',
      enabled: false
    }
  }
  return { version: 1, activeProvider: 'volcengine', providers }
}

function loadProvidersStore(): ProvidersStoreShape {
  const fp = providersStorePath()
  try {
    if (!fs.existsSync(fp)) return defaultStore()
    const raw = JSON.parse(fs.readFileSync(fp, 'utf8')) as ProvidersStoreShape
    const def = defaultStore()
    const merged: ProvidersStoreShape = {
      version: 1,
      activeProvider: raw.activeProvider || def.activeProvider,
      providers: {}
    }
    for (const id of Object.keys(def.providers)) {
      merged.providers[id] = { ...def.providers[id], ...(raw.providers?.[id] || {}) }
    }
    return merged
  } catch {
    return defaultStore()
  }
}

function saveProvidersStore(store: ProvidersStoreShape) {
  fs.writeFileSync(providersStorePath(), JSON.stringify(store, null, 2), { mode: 0o600 })
}

function maskProvider(cfg: ProviderConfig): ProviderConfig {
  return { ...cfg, apiKey: cfg.apiKey ? maskSecret(cfg.apiKey) : '' }
}

function maskStore(s: ProvidersStoreShape) {
  const providers: Record<string, ProviderConfig> = {}
  for (const [k, v] of Object.entries(s.providers)) providers[k] = maskProvider(v)
  return { ...s, providers }
}

/**
 * 通用 OpenAI 兼容 ChatCompletions 客户端（SSE 流式 + 非流式 fallback）
 * 特别为「火山 Ark」兼容：model 字段原样透传（用户配什么就发什么，通常是 ep-xxx 接入点ID）
 */
async function chatCompletions(
  cfg: ProviderConfig,
  payload: { messages: any[]; temperature?: number; max_tokens?: number; stream?: boolean },
  opts?: { signal?: AbortSignal; onDelta?: (chunkText: string) => void }
): Promise<{ ok: boolean; status: number; content: string; error?: string; raw?: any }> {
  if (!cfg.apiKey) return { ok: false, status: 401, content: '', error: `缺少 API Key，请在「设置 → 模型服务商」中为该厂商填写。` }
  if (!cfg.model) return { ok: false, status: 400, content: '', error: `缺少模型 / 接入点 ID 配置。` }

  const baseUrl = cfg.baseUrl.replace(/\/$/, '')
  const url = `${baseUrl}/chat/completions`
  const stream = payload.stream !== false
  const body: any = {
    model: cfg.model,
    messages: payload.messages,
    stream,
    temperature: payload.temperature ?? cfg.temperature ?? 0.7,
    max_tokens: payload.max_tokens ?? cfg.maxTokens ?? 2048
  }

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${cfg.apiKey}`
      },
      body: JSON.stringify(body),
      signal: opts?.signal
    })

    if (!stream) {
      const text = await resp.text()
      let data: any
      try { data = JSON.parse(text) } catch { return { ok: resp.ok, status: resp.status, content: text, error: resp.ok ? undefined : text } }
      const content = data?.choices?.[0]?.message?.content ?? (typeof data === 'string' ? data : JSON.stringify(data))
      return { ok: resp.ok, status: resp.status, content, raw: data, error: resp.ok ? undefined : data?.error?.message || text.slice(0, 300) }
    }

    // ---------- SSE 流式解析（兼容 CRLF / LF / CR，处理尾部 buffer，合并多行 data） ----------
    if (!resp.ok || !resp.body) {
      const errText = await resp.text()
      let msg = `HTTP ${resp.status}`
      try { msg = JSON.parse(errText)?.error?.message || errText.slice(0, 300) } catch {}
      return { ok: false, status: resp.status, content: '', error: msg }
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    let full = ''

    // 将一个 SSE 事件块解析为 delta 文本
    function processEventBlock(block: string) {
      const lines = block.split('\n')
      const dataLines: string[] = []
      for (const line of lines) {
        const s = line.trim()
        if (!s) continue
        if (s.startsWith('data:')) {
          dataLines.push(s.slice(5).trim())
        }
      }
      if (dataLines.length === 0) return
      const payloadStr = dataLines.join('\n')
      if (payloadStr === '[DONE]') return
      try {
        const json = JSON.parse(payloadStr)
        const delta = json?.choices?.[0]?.delta?.content
        if (typeof delta === 'string' && delta) {
          full += delta
          opts?.onDelta?.(delta)
        }
      } catch {
        /* ignore malformed chunk */
      }
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      // 归一化 CRLF -> LF, CR -> LF
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n').replace(/\r/g, '\n')
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 2)
        if (block) processEventBlock(block)
      }
    }
    // 处理尾部残留 buffer
    if (buffer.trim()) processEventBlock(buffer.trim())

    return { ok: true, status: 200, content: full }
  } catch (e: any) {
    if (e?.name === 'AbortError') return { ok: false, status: 0, content: '', error: '请求已取消' }
    return { ok: false, status: 0, content: '', error: e?.message || String(e) }
  }
}

function appendLog(line: string) {
  backendLogs.push(line)
  if (backendLogs.length > MAX_LOG_LINES) {
    backendLogs = backendLogs.slice(-MAX_LOG_LINES)
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:log', line)
  }
}

function broadcastStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:status-change', {
      running: backendRunning,
      port: backendPort
    })
  }
}

function findAvailablePort(startPort: number): Promise<number> {
  return new Promise((resolve) => {
    const server = createServer()
    server.unref()
    server.on('error', () => {
      resolve(findAvailablePort(startPort + 1))
    })
    server.listen(startPort, '127.0.0.1', () => {
      const addr = server.address() as any
      const port = addr.port
      server.close(() => resolve(port))
    })
  })
}

function startBackend() {
  if (backendRunning) return

  const pythonEnv = { ...process.env }
  if (pythonEnv.PYTHONPATH) {
    pythonEnv.PYTHONPATH = repoRoot + path.delimiter + pythonEnv.PYTHONPATH
  } else {
    pythonEnv.PYTHONPATH = repoRoot
  }

  findAvailablePort(8765).then((port) => {
    backendPort = port
    appendLog(`[backend] Starting backend on port ${port}...`)

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'
    backendProcess = spawn(
      pythonCmd,
      ['-m', 'sky_v1.cli.main', 'serve', '--host', '127.0.0.1', '--port', String(port)],
      {
        cwd: repoRoot,
        env: pythonEnv
      }
    )

    backendRunning = false  // 等健康检查通过后再置 true
    broadcastStatus()

    // 轮询 /health 直到后端就绪
    const checkReady = async () => {
      try {
        const resp = await fetch(`http://127.0.0.1:${port}/health`)
        if (resp.ok) {
          backendRunning = true
          broadcastStatus()
          appendLog('[backend] Backend ready (health check passed)')
          return
        }
      } catch {}
      setTimeout(checkReady, 500)
    }
    setTimeout(checkReady, 1000)

    backendProcess.stdout?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n')
      lines.forEach((l) => l && appendLog(`[stdout] ${l}`))
    })

    backendProcess.stderr?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n')
      lines.forEach((l) => l && appendLog(`[stderr] ${l}`))
    })

    backendProcess.on('close', (code: number | null) => {
      appendLog(`[backend] Process exited with code ${code}`)
      backendRunning = false
      backendProcess = null
      broadcastStatus()
    })

    backendProcess.on('error', (err: Error) => {
      appendLog(`[backend] Error: ${err.message}`)
      backendRunning = false
      backendProcess = null
      broadcastStatus()
    })
  })
}

function stopBackend() {
  const proc = backendProcess
  if (proc && !proc.killed) {
    appendLog('[backend] Stopping backend...')
    proc.kill('SIGTERM')
    setTimeout(() => {
      if (proc && !proc.killed) {
        proc.kill('SIGKILL')
      }
    }, 3000)
  }
  // 不在此处清 backendProcess/backendRunning，留给 close 事件处理
}

function restartBackend() {
  stopBackend()
  setTimeout(() => startBackend(), 1000)
}

function setupIpc() {
  ipcMain.handle('backend:status', () => {
    return {
      running: backendRunning,
      port: backendPort,
      logs: backendLogs.slice(-100)
    }
  })

  ipcMain.handle('backend:restart', () => {
    restartBackend()
    return true
  })

  ipcMain.handle('backend:stop', () => {
    stopBackend()
    return true
  })

  ipcMain.handle('app:get-version', () => {
    return app.getVersion()
  })

  ipcMain.handle('app:open-external', (_event, url: string) => {
    return shell.openExternal(url)
  })

  ipcMain.handle('github:cli', async (_event, args: string[]) => {
    return new Promise((resolve) => {
      execFile('gh', args, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
        resolve({
          stdout: stdout || '',
          stderr: stderr || '',
          code: err ? 1 : 0
        })
      })
    })
  })

  ipcMain.handle('api:fetch', async (_event, method: string, urlPath: string, body?: any) => {
    if (!backendRunning) {
      return { ok: false, status: 0, data: null, error: 'Backend is not running. Please wait or click Restart.' }
    }
    try {
      const url = `http://127.0.0.1:${backendPort}/v1${urlPath}`
      const init: RequestInit = {
        method,
        headers: { 'Content-Type': 'application/json' }
      }
      if (body !== undefined && method !== 'GET') {
        init.body = JSON.stringify(body)
      }
      const resp = await fetch(url, init)
      const text = await resp.text()
      try {
        return { ok: resp.ok, status: resp.status, data: JSON.parse(text) }
      } catch {
        return { ok: resp.ok, status: resp.status, data: text }
      }
    } catch (e: any) {
      return { ok: false, status: 0, data: null, error: e.message }
    }
  })

  ipcMain.handle('rag:list-files', async () => {
    try {
      const presetsDir = path.join(repoRoot, 'sky_v1', 'rag', 'presets')
      if (!fs.existsSync(presetsDir)) {
        return { ok: true, files: [] }
      }
      const entries = fs.readdirSync(presetsDir, { withFileTypes: true })
      const files = entries
        .filter((e) => e.isFile())
        .map((e) => e.name)
      return { ok: true, files }
    } catch (e: any) {
      return { ok: false, error: e.message, files: [] }
    }
  })

  ipcMain.handle('rag:ingest', async (_event, filename: string) => {
    try {
      const filePath = path.join(repoRoot, 'sky_v1', 'rag', 'presets', filename)
      const resp = await fetch(`http://127.0.0.1:${backendPort}/v1/rag/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath })
      })
      const text = await resp.text()
      try {
        return { ok: resp.ok, data: JSON.parse(text) }
      } catch {
        return { ok: resp.ok, data: text }
      }
    } catch (e: any) {
      return { ok: false, error: e.message }
    }
  })

  ipcMain.handle('terminal:spawn', (_event, shell?: string, cols: number = 80, rows: number = 24) => {
    if (ptyProcess) {
      ptyProcess.kill()
      ptyProcess = null
    }
    const defaultShell = process.platform === 'win32' ? 'powershell.exe' : process.env.SHELL || 'bash'
    const useShell = shell || defaultShell
    try {
      ptyProcess = ptySpawn(useShell, [], {
        name: 'xterm-256color',
        cols,
        rows,
        cwd: repoRoot,
        env: process.env as any
      })
      ptyProcess.onData((data) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('terminal:data', data)
        }
      })
      ptyProcess.onExit((ev) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('terminal:exit', ev)
        }
        ptyProcess = null
      })
      return { ok: true, pid: ptyProcess.pid }
    } catch (e: any) {
      return { ok: false, error: e.message }
    }
  })

  ipcMain.handle('terminal:input', (_event, data: string) => {
    if (ptyProcess) {
      ptyProcess.write(data)
      return true
    }
    return false
  })

  ipcMain.handle('terminal:kill', () => {
    if (ptyProcess) {
      ptyProcess.kill()
      ptyProcess = null
      return true
    }
    return false
  })

  ipcMain.handle('terminal:resize', (_event, cols: number, rows: number) => {
    if (ptyProcess) {
      try {
        ptyProcess.resize(cols, rows)
        return true
      } catch {
        return false
      }
    }
    return false
  })

  // ===== Providers =====
  ipcMain.handle('providers:list-meta', () => JSON.parse(JSON.stringify(PROVIDER_META)))

  ipcMain.handle('providers:get-store', () => maskStore(loadProvidersStore()))

  ipcMain.handle('providers:update', (_event, providerId: string, patch: Partial<ProviderConfig>) => {
    const meta = PROVIDER_META.find((m) => m.id === providerId)
    if (!meta) return { ok: false, error: `Unknown provider: ${providerId}` }
    const store = loadProvidersStore()
    const prev = store.providers[providerId] || defaultStore().providers[providerId]
    const next: ProviderConfig = { ...prev, ...patch, id: providerId }
    // 掩码值视为「未修改」，保留原值
    if (next.apiKey && /^\S*\*\*\*\S*$/.test(next.apiKey)) next.apiKey = prev.apiKey
    store.providers[providerId] = next
    saveProvidersStore(store)
    return { ok: true, store: maskStore(store) }
  })

  ipcMain.handle('providers:set-active', (_event, providerId: string) => {
    const meta = PROVIDER_META.find((m) => m.id === providerId)
    if (!meta) return { ok: false, error: `Unknown provider: ${providerId}` }
    const store = loadProvidersStore()
    store.activeProvider = providerId
    if (store.providers[providerId]) store.providers[providerId].enabled = true
    saveProvidersStore(store)
    return { ok: true, store: maskStore(store) }
  })

  ipcMain.handle('providers:remove-key', (_event, providerId: string) => {
    const store = loadProvidersStore()
    if (store.providers[providerId]) store.providers[providerId].apiKey = ''
    saveProvidersStore(store)
    return { ok: true, store: maskStore(store) }
  })

  ipcMain.handle('providers:test-connection', async (_event, providerId: string) => {
    const store = loadProvidersStore()
    const cfg = store.providers[providerId]
    if (!cfg) return { ok: false, error: `Unknown provider: ${providerId}` }
    const startedAt = Date.now()
    const res = await chatCompletions(
      cfg,
      {
        messages: [{ role: 'user', content: '你好，请只回复一个字：好' }],
        temperature: 0,
        max_tokens: 8,
        stream: false
      },
      { signal: AbortSignal.timeout(15000) }
    )
    const latency = Date.now() - startedAt
    return {
      ok: res.ok,
      status: res.status,
      latencyMs: latency,
      error: res.error,
      sample: res.content ? res.content.slice(0, 60) : ''
    }
  })

  // 流式聊天（给 ChatView 用）
  ipcMain.handle(
    'providers:chat',
    async (
      event,
      args: { providerId?: string; messages: any[]; temperature?: number; max_tokens?: number; stream?: boolean }
    ) => {
      const store = loadProvidersStore()
      const pid = args.providerId || store.activeProvider
      const cfg = store.providers[pid]
      if (!cfg) return { ok: false, status: 0, content: '', error: `未找到 provider: ${pid}` }
      const useStream = args.stream !== false
      const win = BrowserWindow.fromWebContents(event.sender)
      if (!useStream) {
        const res = await chatCompletions(cfg, args, { signal: AbortSignal.timeout(60000) })
        return { ok: res.ok, status: res.status, content: res.content, error: res.error, providerId: pid }
      }
      // 流式：经 SSE 解析后通过事件逐块推送到渲染层，最后同步返回完整结果
      const full = await new Promise<string>((resolve, reject) => {
        let acc = ''
        chatCompletions(
          cfg,
          args,
          {
            signal: AbortSignal.timeout(120000),
            onDelta: (delta) => {
              acc += delta
              if (win && !win.isDestroyed()) {
                win.webContents.send('providers:chat:delta', { providerId: pid, delta })
              }
            }
          }
        ).then(
          (res) => {
            if (!res.ok) reject(new Error(res.error || `HTTP ${res.status}`))
            else resolve(res.content || acc)
          },
          (err) => reject(err)
        )
      })
      return { ok: true, status: 200, content: full, providerId: pid }
    }
  )

  // =========================================================================
  // 媒体生成：生图 / 生视频 / 3D 模型描述（生图生视频：MiniMax + 字节火山；3D：字节推理流）
  // =========================================================================

  // ----- 生图 -----
  ipcMain.handle(
    'media:generate-image',
    async (
      _event,
      args: {
        provider: 'minimax' | 'volcengine'
        apiKey: string
        prompt: string
        model?: string
        size?: string // '1024x1024' (volc) or '1:1'/'16:9' (minimax aspect_ratio)
        n?: number
      }
    ) => {
      if (!args.apiKey) return { ok: false, error: '缺少 API Key' }
      if (!args.prompt?.trim()) return { ok: false, error: '缺少 prompt' }
      try {
        if (args.provider === 'minimax') {
          // MiniMax 文生图：POST https://api.minimax.io/v1/image_generation （模型 image-01，2026 仍为最新）
          const url = 'https://api.minimax.io/v1/image_generation'
          const body = {
            model: args.model || 'image-01',
            prompt: args.prompt,
            aspect_ratio: args.size || '1:1',
            response_format: 'url',
            n: args.n || 1,
            prompt_optimizer: true
          }
          const resp = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${args.apiKey}`
            },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(120000)
          })
          const text = await resp.text()
          let data: any
          try { data = JSON.parse(text) } catch { return { ok: false, status: resp.status, error: text.slice(0, 500) } }
          if (!resp.ok) return { ok: false, status: resp.status, error: data?.base_resp?.status_msg || data?.message || text.slice(0, 300) }
          const urls: string[] = data?.data?.image_urls || []
          return { ok: true, status: 200, images: urls, raw: data }
        } else {
          // 字节火山方舟 文生图（Doubao Seedream 5.0 Pro）：POST /api/v3/images/generations
          const baseUrl = 'https://ark.cn-beijing.volces.com/api/v3'
          const url = `${baseUrl}/images/generations`
          const body = {
            model: args.model || 'doubao-seedream-5-0-pro-260628',
            prompt: args.prompt,
            size: args.size || '1024x1024',
            response_format: 'url',
            n: args.n || 1
          }
          const resp = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${args.apiKey}`
            },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(120000)
          })
          const text = await resp.text()
          let data: any
          try { data = JSON.parse(text) } catch { return { ok: false, status: resp.status, error: text.slice(0, 500) } }
          if (!resp.ok) return { ok: false, status: resp.status, error: data?.error?.message || text.slice(0, 300) }
          const urls: string[] = (data?.data || []).map((d: any) => d.url).filter(Boolean)
          return { ok: true, status: 200, images: urls, raw: data }
        }
      } catch (e: any) {
        if (e?.name === 'AbortError') return { ok: false, error: '请求超时（120s）' }
        return { ok: false, error: e?.message || String(e) }
      }
    }
  )

  // ----- 生视频（异步轮询）-----
  ipcMain.handle(
    'media:generate-video',
    async (
      event,
      args: {
        provider: 'minimax' | 'volcengine'
        apiKey: string
        prompt: string
        model?: string
        duration?: number // 秒
      }
    ) => {
      if (!args.apiKey) return { ok: false, error: '缺少 API Key' }
      if (!args.prompt?.trim()) return { ok: false, error: '缺少 prompt' }
      const win = BrowserWindow.fromWebContents(event.sender)
      const sendProgress = (p: { phase: string; message?: string; progress?: number }) => {
        if (win && !win.isDestroyed()) {
          win.webContents.send('media:video:progress', p)
        }
      }
      try {
        let taskId = ''
        if (args.provider === 'minimax') {
          // MiniMax 文生视频：MiniMax-H3（2026-08-03 开源最新版，原生 30s/2K/立体声）
          // 接口：POST https://api.minimax.io/v1/video_generation，content[] 多模态结构
          sendProgress({ phase: 'submitting', message: '提交视频生成任务…' })
          const url = 'https://api.minimax.io/v1/video_generation'
          const duration = args.duration || 6
          const body = {
            model: args.model || 'MiniMax-H3',
            content: [{ type: 'text', text: args.prompt }],
            duration
          }
          const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${args.apiKey}` },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(30000)
          })
          const text = await resp.text()
          let data: any
          try { data = JSON.parse(text) } catch { return { ok: false, error: text.slice(0, 500) } }
          if (!resp.ok) return { ok: false, status: resp.status, error: data?.base_resp?.status_msg || text.slice(0, 300) }
          taskId = data?.task_id
          if (!taskId) return { ok: false, error: '未获取到 task_id' }

          // 轮询：H3 返回 content.url，旧版返回 file.download_url
          sendProgress({ phase: 'generating', message: '视频生成中…' })
          const pollUrl = `https://api.minimax.io/v1/query/video_generation?task_id=${taskId}`
          for (let i = 0; i < 120; i++) {
            await new Promise((r) => setTimeout(r, 5000))
            sendProgress({ phase: 'polling', progress: i, message: `轮询中 (${i + 1}/120)…` })
            const pr = await fetch(pollUrl, { headers: { Authorization: `Bearer ${args.apiKey}` } })
            const pt = await pr.text()
            let pd: any
            try { pd = JSON.parse(pt) } catch { continue }
            const status = pd?.status || pd?.file?.download_status
            if (status === 'Success' || status === 'success' || status === 'Succeeded' || status === 'succeeded') {
              // H3 用 content.url，旧版用 file.download_url
              const videoUrl = pd?.content?.url || pd?.file?.download_url
              if (videoUrl) return { ok: true, status: 200, video: videoUrl, taskId, raw: pd }
            }
            if (status === 'Failed' || status === 'failed') {
              return { ok: false, error: pd?.base_resp?.status_msg || '视频生成失败' }
            }
          }
          return { ok: false, error: '视频生成超时（10分钟）' }
        } else {
          // 字节火山方舟 文生视频（Doubao Seedance 2.0，方舟稳定版）
          // Seedance 2.5 仅 BytePlus 海外灰度，方舟暂未 GA
          sendProgress({ phase: 'submitting', message: '提交视频生成任务…' })
          const url = 'https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks'
          const body = {
            model: args.model || 'doubao-seedance-2-0-260128',
            content: [{ type: 'text', text: args.prompt }]
          }
          const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${args.apiKey}` },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(30000)
          })
          const text = await resp.text()
          let data: any
          try { data = JSON.parse(text) } catch { return { ok: false, error: text.slice(0, 500) } }
          if (!resp.ok) return { ok: false, status: resp.status, error: data?.error?.message || text.slice(0, 300) }
          taskId = data?.id
          if (!taskId) return { ok: false, error: '未获取到 task_id' }

          sendProgress({ phase: 'generating', message: '视频生成中…' })
          const pollUrl = `${url}/${taskId}`
          for (let i = 0; i < 120; i++) {
            await new Promise((r) => setTimeout(r, 5000))
            sendProgress({ phase: 'polling', progress: i, message: `轮询中 (${i + 1}/120)…` })
            const pr = await fetch(pollUrl, { headers: { Authorization: `Bearer ${args.apiKey}` } })
            const pt = await pr.text()
            let pd: any
            try { pd = JSON.parse(pt) } catch { continue }
            const status = pd?.status
            if (status === 'succeeded') {
              // Seedance 2.0 content[].video_url 或 content.video_url
              const videoUrl = pd?.content?.video_url
                || (Array.isArray(pd?.content) ? pd.content.find((c: any) => c?.video_url)?.video_url : undefined)
              if (videoUrl) return { ok: true, status: 200, video: videoUrl, taskId, raw: pd }
            }
            if (status === 'failed') {
              return { ok: false, error: pd?.error?.message || '视频生成失败' }
            }
          }
          return { ok: false, error: '视频生成超时（10分钟）' }
        }
      } catch (e: any) {
        if (e?.name === 'AbortError') return { ok: false, error: '请求超时' }
        return { ok: false, error: e?.message || String(e) }
      }
    }
  )

  // ----- 3D 模型生成（字节在线推理流 → 生成场景 JSON → 前端 Three.js 渲染）-----
  ipcMain.handle(
    'media:generate-3d',
    async (
      event,
      args: {
        apiKey: string
        prompt: string
        model: string // 火山 Ark endpoint ID
        baseUrl?: string
      }
    ) => {
      if (!args.apiKey) return { ok: false, error: '缺少火山 API Key' }
      if (!args.model) return { ok: false, error: '缺少火山推理接入点 Endpoint ID' }
      if (!args.prompt?.trim()) return { ok: false, error: '缺少 prompt' }
      const win = BrowserWindow.fromWebContents(event.sender)

      const sysPrompt = `你是一个 3D 场景生成助手。用户会给你一段描述，你需要输出一个 JSON 格式的 Three.js 场景描述。
JSON 格式如下：
{
  "objects": [
    {
      "type": "box" | "sphere" | "cylinder" | "cone" | "torus" | "plane" | "dodecahedron" | "icosahedron" | "octahedron" | "tetrahedron" | "torusknot",
      "position": [x, y, z],
      "rotation": [x, y, z],
      "scale": [x, y, z],
      "color": "#hexcolor",
      "metalness": 0.0-1.0,
      "roughness": 0.0-1.0,
      "wireframe": false
    }
  ],
  "background": "#hexcolor",
  "ground": true | false,
  "groundColor": "#hexcolor"
}

要求：
- 只输出 JSON，不要任何解释文字
- 坐标范围 -5 到 5
- 颜色用 #rrggbb 格式
- 根据用户描述合理摆放物体
- 最多 15 个物体`

      try {
        const baseUrl = (args.baseUrl || 'https://ark.cn-beijing.volces.com/api/v3').replace(/\/$/, '')
        const url = `${baseUrl}/chat/completions`
        const body = {
          model: args.model,
          messages: [
            { role: 'system', content: sysPrompt },
            { role: 'user', content: args.prompt }
          ],
          stream: true,
          temperature: 0.7,
          max_tokens: 2048
        }

        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${args.apiKey}` },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(120000)
        })

        if (!resp.ok || !resp.body) {
          const errText = await resp.text()
          let msg = `HTTP ${resp.status}`
          try { msg = JSON.parse(errText)?.error?.message || errText.slice(0, 300) } catch {}
          return { ok: false, status: resp.status, error: msg }
        }

        const reader = resp.body.getReader()
        const decoder = new TextDecoder('utf-8')
        let buffer = ''
        let full = ''

        const processBlock = (block: string) => {
          const lines = block.split('\n')
          const dataLines: string[] = []
          for (const line of lines) {
            const s = line.trim()
            if (!s) continue
            if (s.startsWith('data:')) dataLines.push(s.slice(5).trim())
          }
          if (dataLines.length === 0) return
          const payloadStr = dataLines.join('\n')
          if (payloadStr === '[DONE]') return
          try {
            const json = JSON.parse(payloadStr)
            const delta = json?.choices?.[0]?.delta?.content
            if (typeof delta === 'string' && delta) {
              full += delta
              if (win && !win.isDestroyed()) {
                win.webContents.send('media:3d:delta', { delta, full })
              }
            }
          } catch { /* ignore */ }
        }

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n').replace(/\r/g, '\n')
          let idx: number
          while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const block = buffer.slice(0, idx).trim()
            buffer = buffer.slice(idx + 2)
            if (block) processBlock(block)
          }
        }
        if (buffer.trim()) processBlock(buffer.trim())

        // 尝试从流式输出中提取 JSON
        let sceneJson: any = null
        try {
          // 找到第一个 { 和最后一个 }
          const start = full.indexOf('{')
          const end = full.lastIndexOf('}')
          if (start >= 0 && end > start) {
            sceneJson = JSON.parse(full.slice(start, end + 1))
          }
        } catch {
          // 尝试修复常见 JSON 问题
          try {
            const start = full.indexOf('{')
            const end = full.lastIndexOf('}')
            if (start >= 0 && end > start) {
              const candidate = full.slice(start, end + 1).replace(/,\s*}/g, '}').replace(/,\s*]/g, ']')
              sceneJson = JSON.parse(candidate)
            }
          } catch { /* JSON 解析失败 */ }
        }

        return { ok: true, status: 200, content: full, scene: sceneJson }
      } catch (e: any) {
        if (e?.name === 'AbortError') return { ok: false, error: '请求超时（120s）' }
        return { ok: false, error: e?.message || String(e) }
      }
    }
  )
}

function setupAutoUpdater() {
  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = true

  autoUpdater.on('update-available', (info) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-available', info)
    }
  })

  autoUpdater.on('update-not-available', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-not-available')
    }
  })

  autoUpdater.on('download-progress', (progress) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:download-progress', progress)
    }
  })

  autoUpdater.on('update-downloaded', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-downloaded')
    }
  })

  autoUpdater.on('error', (err) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-error', err.message)
    }
  })

  if (!app.isPackaged) return

  try {
    autoUpdater.checkForUpdates().catch(() => {})
  } catch {}
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: 'Sky V1 Omni Desktop',
    icon: path.join(__dirname, '..', '..', 'build', 'icon.png'),
    webPreferences: {
      preload: path.resolve(__dirname, '..', 'preload', 'index.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true
    },
    backgroundColor: '#1e1e1e',
    show: false
  })

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })

  if (!app.isPackaged && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(path.resolve(__dirname, '..', 'renderer', 'index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.whenReady().then(() => {
  setupIpc()
  createWindow()
  setupAutoUpdater()
  startBackend()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
    if (!backendRunning) {
      startBackend()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    stopBackend()
    if (ptyProcess) {
      ptyProcess.kill()
      ptyProcess = null
    }
    app.quit()
  }
  // macOS: 保留后端进程，只关窗口
})

app.on('before-quit', () => {
  stopBackend()
  if (ptyProcess) {
    ptyProcess.kill()
    ptyProcess = null
  }
})
