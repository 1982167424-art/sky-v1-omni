import { app, BrowserWindow, ipcMain, shell } from 'electron'
import { spawn, exec, ChildProcess } from 'child_process'
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
    id: 'xiaomi',
    name: '小米 MiLM',
    brandColor: '#FF6900',
    defaultBaseUrl: 'https://api.minimimax.ai/v1',
    docsUrl: 'https://www.xiaomi.cn/milm',
    modelLabel: '模型 ID',
    modelPlaceholder: '例如 minimax-abab6.5s-chat',
    supportsStreaming: true,
    defaultModel: 'minimax-abab6.5s-chat'
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

    // ---------- SSE 流式解析 ----------
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

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 2)
        if (!block) continue
        const lines = block.split('\n')
        for (const line of lines) {
          const s = line.trim()
          if (!s.startsWith('data:')) continue
          const payloadStr = s.slice(5).trim()
          if (!payloadStr) continue
          if (payloadStr === '[DONE]') continue
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
      }
    }

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

    backendRunning = true
    broadcastStatus()

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
  if (backendProcess && !backendProcess.killed) {
    appendLog('[backend] Stopping backend...')
    backendProcess.kill('SIGTERM')
    setTimeout(() => {
      if (backendProcess && !backendProcess.killed) {
        backendProcess.kill('SIGKILL')
      }
    }, 3000)
  }
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
      const cmd = `gh ${args.join(' ')}`
      exec(cmd, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
        resolve({
          stdout: stdout || '',
          stderr: stderr || '',
          code: err?.code ?? 0
        })
      })
    })
  })

  ipcMain.handle('api:fetch', async (_event, method: string, urlPath: string, body?: any) => {
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
        const res = await chatCompletions(cfg, args)
        return { ok: res.ok, status: res.status, content: res.content, error: res.error, providerId: pid }
      }
      // 流式：经 SSE 解析后通过事件逐块推送到渲染层，最后同步返回完整结果
      const full = await new Promise<string>((resolve, reject) => {
        let acc = ''
        chatCompletions(
          cfg,
          args,
          {
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
  })
})

app.on('window-all-closed', () => {
  stopBackend()
  if (ptyProcess) {
    ptyProcess.kill()
    ptyProcess = null
  }
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopBackend()
  if (ptyProcess) {
    ptyProcess.kill()
    ptyProcess = null
  }
})
