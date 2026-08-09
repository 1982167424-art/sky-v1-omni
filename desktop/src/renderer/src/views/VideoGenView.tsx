import React, { useState, useEffect, useRef } from 'react'
import { Video as VideoIcon, Loader2, Download, Server, ChevronDown, CheckCircle, ExternalLink, Film } from 'lucide-react'
import type { ProviderConfig, ProvidersStore } from '../../preload/index'
import type { VideoProgress } from '../../preload/index'

type MediaProvider = 'minimax' | 'volcengine'

export default function VideoGenView() {
  const [prompt, setPrompt] = useState('')
  const [provider, setProvider] = useState<MediaProvider>('volcengine')
  const [model, setModel] = useState('')
  const [duration, setDuration] = useState(6)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<VideoProgress | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [store, setStore] = useState<ProvidersStore | null>(null)
  const [minimaxKey, setMinimaxKey] = useState('')
  const [minimaxKeyDraft, setMinimaxKeyDraft] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    window.sky.providers.getStore().then((s) => {
      setStore(s)
    }).catch(() => {})
    const off = window.sky.media.onVideoProgress((p) => setProgress(p))
    return () => { off() }
  }, [])

  useEffect(() => {
    try {
      const saved = localStorage.getItem('sky:minimax-key')
      if (saved) setMinimaxKey(saved)
    } catch {}
  }, [])

  const volcCfg: ProviderConfig | undefined = store?.providers?.volcengine
  const volcReady = !!(volcCfg?.apiKey && volcCfg?.model)
  const minimaxReady = !!(minimaxKey && minimaxKey.trim())

  const currentModel = () => {
    if (provider === 'minimax') return model || 'MiniMax-H3'
    return model || 'doubao-seedance-2-0-260128'
  }

  const currentKey = () => {
    if (provider === 'minimax') return minimaxKey
    return volcCfg?.apiKey || ''
  }

  const generate = async () => {
    if (!prompt.trim() || loading) return
    const key = currentKey()
    if (!key) {
      setError(provider === 'minimax'
        ? '请输入 MiniMax API Key'
        : '请先在「设置 → 模型服务商 → 火山引擎方舟」配置 API Key')
      return
    }
    setLoading(true)
    setError(null)
    setVideoUrl(null)
    setProgress({ phase: 'starting', message: '准备提交任务…' })
    try {
      const result = await window.sky.media.generateVideo({
        provider,
        apiKey: key,
        prompt: prompt.trim(),
        model: currentModel(),
        duration
      })
      if (result.ok && result.video) {
        setVideoUrl(result.video)
      } else {
        setError(result.error || '视频生成失败')
      }
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setLoading(false)
      setProgress(null)
    }
  }

  const saveMinimaxKey = () => {
    const v = minimaxKeyDraft.trim()
    setMinimaxKey(v)
    try { localStorage.setItem('sky:minimax-key', v) } catch {}
    setMinimaxKeyDraft('')
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: 16, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <VideoIcon size={20} color="var(--fg-accent)" />
          <span style={{ fontWeight: 600 }}>Video Generation</span>
          <span style={{ fontSize: 11, color: 'var(--fg-secondary)', marginLeft: 8 }}>
            MiniMax H3 + 字节火山方舟（豆包 Seedance 2.0）· 异步轮询
          </span>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
            <Server size={13} />
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>服务商：</span>
            <select
              value={provider}
              onChange={(e) => { setProvider(e.target.value as MediaProvider); setModel('') }}
              style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
            >
              <option value="volcengine">字节火山方舟（Seedance 2.0）</option>
              <option value="minimax">MiniMax（H3）</option>
            </select>
            <ChevronDown size={12} />
          </div>

          {provider === 'minimax' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
              <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>时长(秒)：</span>
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
              >
                {[4, 5, 6, 8, 10, 12, 15].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>模型：</span>
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={provider === 'minimax' ? 'MiniMax-H3' : 'doubao-seedance-2-0-260128'}
              style={{ width: 240, background: 'transparent', border: 'none', outline: 'none', fontSize: 12 }}
            />
          </div>

          <div style={{ flex: 1 }} />

          {provider === 'volcengine' ? (
            volcReady ? (
              <span style={{ color: 'var(--success)', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
                <CheckCircle size={12} /> 火山已配置
              </span>
            ) : (
              <span style={{ color: 'var(--warning)', fontSize: 12 }}>需先配置火山 API Key</span>
            )
          ) : (
            minimaxReady ? (
              <span style={{ color: 'var(--success)', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
                <CheckCircle size={12} /> MiniMax 已配置
              </span>
            ) : (
              <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                <input
                  type="password"
                  value={minimaxKeyDraft}
                  onChange={(e) => setMinimaxKeyDraft(e.target.value)}
                  placeholder="粘贴 MiniMax API Key"
                  style={{ width: 200, fontSize: 12 }}
                />
                <button onClick={saveMinimaxKey} style={{ padding: '4px 10px', fontSize: 11 }}>保存</button>
              </div>
            )
          )}
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate() } }}
            placeholder="描述你想生成的视频... (Ctrl+Enter 生成)"
            rows={2}
            style={{ flex: 1, minHeight: 52, padding: '10px 12px' }}
          />
          <button
            onClick={generate}
            disabled={loading || !prompt.trim()}
            style={{ height: 52, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {loading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Film size={16} />}
            {loading ? '生成中…' : '生成视频'}
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {error && (
          <div style={{ padding: 12, borderRadius: 4, background: 'rgba(244,135,113,0.1)', border: '1px solid var(--error)', color: 'var(--error)', marginBottom: 16 }}>
            {error}
          </div>
        )}
        {loading && progress && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--fg-secondary)', marginBottom: 8 }}>
              <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} />
              <span style={{ fontWeight: 500 }}>{progress.message || '处理中…'}</span>
            </div>
            <div style={{ padding: '10px 14px', background: 'var(--bg-sidebar)', border: '1px solid var(--border-color)', borderRadius: 4, fontSize: 12 }}>
              <div>阶段：<code style={{ color: 'var(--fg-accent)' }}>{progress.phase}</code></div>
              <div style={{ color: 'var(--fg-secondary)', marginTop: 4 }}>
                视频生成通常需要 1-5 分钟，请耐心等待。后台每 5 秒轮询一次任务状态。
              </div>
            </div>
          </div>
        )}
        {videoUrl && (
          <div style={{ maxWidth: 900 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
              <CheckCircle size={16} color="var(--success)" /> 视频生成成功
            </div>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              autoPlay
              style={{ width: '100%', maxHeight: 500, background: '#000', borderRadius: 6, border: '1px solid var(--border-color)' }}
            />
            <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
              <button onClick={() => window.sky.app.openExternal(videoUrl)} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <ExternalLink size={14} /> 新窗口打开
              </button>
              <a href={videoUrl} download style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', background: 'var(--bg-input)', color: 'var(--fg-primary)', border: '1px solid var(--border-color)', borderRadius: 4, textDecoration: 'none', fontSize: 13 }}>
                <Download size={14} /> 下载视频
              </a>
            </div>
          </div>
        )}
        {!videoUrl && !loading && !error && (
          <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60, lineHeight: 1.7 }}>
            输入 Prompt 并选择服务商，点击「生成视频」开始创作。<br />
            支持 <strong style={{ color: 'var(--fg-primary)' }}>MiniMax H3</strong>（原生 30s/2K/立体声）和 <strong style={{ color: 'var(--fg-primary)' }}>字节火山豆包 Seedance 2.0</strong> 双引擎。<br />
            ⚠️ 视频生成为异步任务，需轮询等待（通常 1-5 分钟）。
          </div>
        )}
      </div>
    </div>
  )
}
