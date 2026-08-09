import { contextBridge, ipcRenderer } from 'electron'

export interface BackendStatus {
  running: boolean
  port: number
  logs: string[]
}

export interface BackendApi {
  status: () => Promise<BackendStatus>
  restart: () => Promise<boolean>
  stop: () => Promise<boolean>
  onLog: (cb: (line: string) => void) => () => void
  onStatusChange: (cb: (status: { running: boolean; port: number }) => void) => () => void
}

export interface AppApi {
  getVersion: () => Promise<string>
  openExternal: (url: string) => Promise<void>
}

export interface GithubCliResult {
  stdout: string
  stderr: string
  code: number
}

export interface GithubApi {
  cli: (args: string[]) => Promise<GithubCliResult>
  getReleaseNotes: (tag?: string) => Promise<any>
}

export interface TerminalApi {
  spawn: (shell?: string, cols?: number, rows?: number) => Promise<{ ok: boolean; pid?: number; error?: string }>
  input: (data: string) => Promise<boolean>
  kill: () => Promise<boolean>
  resize: (cols: number, rows: number) => Promise<boolean>
  onData: (cb: (data: string) => void) => () => void
  onExit: (cb: (ev: any) => void) => () => void
}

export interface ApiResult {
  ok: boolean
  status: number
  data: any
  error?: string
}

export interface RestApi {
  get: (url: string) => Promise<ApiResult>
  post: (url: string, body?: any) => Promise<ApiResult>
  put: (url: string, body?: any) => Promise<ApiResult>
  del: (url: string) => Promise<ApiResult>
}

export interface RagApi {
  listFiles: () => Promise<{ ok: boolean; files: string[]; error?: string }>
  ingest: (filename: string) => Promise<{ ok: boolean; data?: any; error?: string }>
}

export interface ProviderMeta {
  id: string
  name: string
  brandColor: string
  defaultBaseUrl: string
  docsUrl: string
  modelPlaceholder: string
  modelLabel: string
  supportsStreaming: boolean
  defaultModel?: string
}

export interface ProviderConfig {
  id: string
  apiKey: string
  baseUrl: string
  model: string
  enabled: boolean
  temperature?: number
  maxTokens?: number
}

export interface ProvidersStore {
  version: 1
  activeProvider: string
  providers: Record<string, ProviderConfig>
}

export interface ChatResult {
  ok: boolean
  status: number
  content: string
  error?: string
  providerId?: string
}

export interface ProvidersApi {
  listMeta: () => Promise<ProviderMeta[]>
  getStore: () => Promise<ProvidersStore>
  update: (
    providerId: string,
    patch: Partial<ProviderConfig>
  ) => Promise<{ ok: boolean; store?: ProvidersStore; error?: string }>
  setActive: (providerId: string) => Promise<{ ok: boolean; store?: ProvidersStore; error?: string }>
  removeKey: (providerId: string) => Promise<{ ok: boolean; store?: ProvidersStore; error?: string }>
  testConnection: (
    providerId: string
  ) => Promise<{ ok: boolean; status: number; latencyMs: number; error?: string; sample?: string }>
  chat: (args: {
    providerId?: string
    messages: { role: string; content: string }[]
    temperature?: number
    max_tokens?: number
    stream?: boolean
  }) => Promise<ChatResult>
  onChatDelta: (cb: (ev: { providerId: string; delta: string }) => void) => () => void
}

export interface SkyWindow {
  backend: BackendApi
  app: AppApi
  github: GithubApi
  terminal: TerminalApi
  api: RestApi
  rag: RagApi
  providers: ProvidersApi
}

const backend: BackendApi = {
  status: () => ipcRenderer.invoke('backend:status'),
  restart: () => ipcRenderer.invoke('backend:restart'),
  stop: () => ipcRenderer.invoke('backend:stop'),
  onLog: (cb) => {
    const listener = (_e: any, line: string) => cb(line)
    ipcRenderer.on('backend:log', listener)
    return () => ipcRenderer.removeListener('backend:log', listener)
  },
  onStatusChange: (cb) => {
    const listener = (_e: any, status: any) => cb(status)
    ipcRenderer.on('backend:status-change', listener)
    return () => ipcRenderer.removeListener('backend:status-change', listener)
  }
}

const appApi: AppApi = {
  getVersion: () => ipcRenderer.invoke('app:get-version'),
  openExternal: (url: string) => ipcRenderer.invoke('app:open-external', url)
}

const github: GithubApi = {
  cli: (args: string[]) => ipcRenderer.invoke('github:cli', args),
  getReleaseNotes: async (tag?: string) => {
    const args = tag ? ['release', 'view', tag, '--json', 'tag_name,name,body,published_at'] : ['release', 'view', '--json', 'tag_name,name,body,published_at']
    const result = await ipcRenderer.invoke('github:cli', args)
    try {
      return result.code === 0 ? JSON.parse(result.stdout) : null
    } catch {
      return null
    }
  }
}

const terminal: TerminalApi = {
  spawn: (shell?, cols?, rows?) => ipcRenderer.invoke('terminal:spawn', shell, cols, rows),
  input: (data: string) => ipcRenderer.invoke('terminal:input', data),
  kill: () => ipcRenderer.invoke('terminal:kill'),
  resize: (cols: number, rows: number) => ipcRenderer.invoke('terminal:resize', cols, rows),
  onData: (cb) => {
    const listener = (_e: any, data: string) => cb(data)
    ipcRenderer.on('terminal:data', listener)
    return () => ipcRenderer.removeListener('terminal:data', listener)
  },
  onExit: (cb) => {
    const listener = (_e: any, ev: any) => cb(ev)
    ipcRenderer.on('terminal:exit', listener)
    return () => ipcRenderer.removeListener('terminal:exit', listener)
  }
}

const api: RestApi = {
  get: (url: string) => ipcRenderer.invoke('api:fetch', 'GET', url),
  post: (url: string, body?: any) => ipcRenderer.invoke('api:fetch', 'POST', url, body),
  put: (url: string, body?: any) => ipcRenderer.invoke('api:fetch', 'PUT', url, body),
  del: (url: string) => ipcRenderer.invoke('api:fetch', 'DELETE', url)
}

const rag: RagApi = {
  listFiles: () => ipcRenderer.invoke('rag:list-files'),
  ingest: (filename: string) => ipcRenderer.invoke('rag:ingest', filename)
}

const providers: ProvidersApi = {
  listMeta: () => ipcRenderer.invoke('providers:list-meta'),
  getStore: () => ipcRenderer.invoke('providers:get-store'),
  update: (providerId, patch) => ipcRenderer.invoke('providers:update', providerId, patch),
  setActive: (providerId) => ipcRenderer.invoke('providers:set-active', providerId),
  removeKey: (providerId) => ipcRenderer.invoke('providers:remove-key', providerId),
  testConnection: (providerId) => ipcRenderer.invoke('providers:test-connection', providerId),
  chat: (args) => ipcRenderer.invoke('providers:chat', args),
  onChatDelta: (cb) => {
    const listener = (_e: any, ev: any) => cb(ev)
    ipcRenderer.on('providers:chat:delta', listener)
    return () => ipcRenderer.removeListener('providers:chat:delta', listener)
  }
}

const sky: SkyWindow = {
  backend,
  app: appApi,
  github,
  terminal,
  api,
  rag,
  providers
}

contextBridge.exposeInMainWorld('sky', sky)
