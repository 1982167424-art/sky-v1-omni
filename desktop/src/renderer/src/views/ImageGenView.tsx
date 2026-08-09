import React, { useState, useEffect } from 'react'
import { Image as ImageIcon, Loader2, Download, Server, ChevronDown, CheckCircle, ExternalLink } from 'lucide-react'
import type { ProviderMeta, ProviderConfig, ProvidersStore } from '../../preload/index'

type MediaProvider = 'minimax' | 'volcengine'

interface ImageItem {
  url: string
  prompt: string
  provider: MediaProvider
  timestamp: number
}

export default function ImageGenView() {
  const [prompt, setPrompt] = useState('')
  const [provider, setProvider] = useState<MediaProvider>('volcengine')
  const [size, setSize] = useState('1024x1024')
  const [n, setN] = useState(1)
  const [model, setModel] = useState('')
  const [loading, setLoading] = useState(false)
  const [images, setImages] = useState<ImageItem[]>([])
  const [error, setError] = useState<string | null>(null)
  const [store, setStore] = useState<ProvidersStore | null>(null)
  const [minimaxKey, setMinimaxKey] = useState('')
  const [minimaxKeyDraft, setMinimaxKeyDraft] = useState('')

  useEffect(() => {
    window.sky.providers.getStore().then((s) => {
      setStore(s)
      // 如果火山已配置就预选
      const vc = s.providers?.volcengine
      if (vc?.apiKey && vc?.model) setProvider('volcengine')
    }).catch(() => {})
  }, [])

  // 读取本地保存的 MiniMax Key
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
    if (provider === 'minimax') return model || 'image-01'
    return model || 'doubao-seedream-5-0-pro-260628'
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
        ? '请输入 MiniMax API Key（保存在本机 localStorage）'
        : '请先在「设置 → 模型服务商 → 火山引擎方舟」配置 API Key')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const result = await window.sky.media.generateImage({
        provider,
        apiKey: key,
        prompt: prompt.trim(),
        model: currentModel(),
        size: provider === 'minimax' ? (size.includes(':') ? size : '1:1') : size,
        n
      })
      if (result.ok && result.images && result.images.length > 0) {
        const newImages: ImageItem[] = result.images.map((url) => ({
          url, prompt: prompt.trim(), provider, timestamp: Date.now()
        }))
        setImages((prev) => [...newImages, ...prev])
      } else {
        setError(result.error || '生成失败，未返回图片')
      }
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setLoading(false)
    }
  }

  const saveMinimaxKey = () => {
    const v = minimaxKeyDraft.trim()
    setMinimaxKey(v)
    try { localStorage.setItem('sky:minimax-key', v) } catch {}
    setMinimaxKeyDraft('')
  }

  const sizeOptions = provider === 'minimax'
    ? ['1:1', '3:4', '4:3', '9:16', '16:9']
    : ['1024x1024', '1024x1792', '1792x1024', '768x1024', '1024x768']

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: 16, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <ImageIcon size={20} color="var(--fg-accent)" />
          <span style={{ fontWeight: 600 }}>Image Generation</span>
          <span style={{ fontSize: 11, color: 'var(--fg-secondary)', marginLeft: 8 }}>
            MiniMax image-01 + 字节火山方舟（豆包 Seedream 5.0 Pro）
          </span>
        </div>

        {/* Provider 选择 */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
            <Server size={13} />
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>服务商：</span>
            <select
              value={provider}
              onChange={(e) => { setProvider(e.target.value as MediaProvider); setModel(''); setSize(provider === 'minimax' ? '1:1' : '1024x1024') }}
              style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
            >
              <option value="volcengine">字节火山方舟（Seedream 5.0 Pro）</option>
              <option value="minimax">MiniMax（image-01）</option>
            </select>
            <ChevronDown size={12} />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>尺寸：</span>
            <select
              value={size}
              onChange={(e) => setSize(e.target.value)}
              style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
            >
              {sizeOptions.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>数量：</span>
            <select
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
            >
              {[1, 2, 3, 4].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>模型：</span>
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={provider === 'minimax' ? 'image-01' : 'doubao-seedream-5-0-pro-260628'}
              style={{ width: 220, background: 'transparent', border: 'none', outline: 'none', fontSize: 12 }}
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
            placeholder="描述你想生成的图片... (Ctrl+Enter 生成)"
            rows={2}
            style={{ flex: 1, minHeight: 52, padding: '10px 12px' }}
          />
          <button
            onClick={generate}
            disabled={loading || !prompt.trim()}
            style={{ height: 52, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {loading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : <ImageIcon size={16} />}
            {loading ? '生成中…' : '生成图片'}
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {error && (
          <div style={{ padding: 12, borderRadius: 4, background: 'rgba(244,135,113,0.1)', border: '1px solid var(--error)', color: 'var(--error)', marginBottom: 16 }}>
            {error}
          </div>
        )}
        {loading && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--fg-secondary)', marginBottom: 16 }}>
            <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} />
            正在通过 {provider === 'minimax' ? 'MiniMax' : '火山方舟'} 生成图片，请稍候…
          </div>
        )}
        {images.length === 0 && !loading && !error && (
          <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60, lineHeight: 1.7 }}>
            输入 Prompt 并选择服务商，点击「生成图片」开始创作。<br />
            支持 <strong style={{ color: 'var(--fg-primary)' }}>MiniMax image-01</strong> 和 <strong style={{ color: 'var(--fg-primary)' }}>字节火山豆包 Seedream 5.0 Pro</strong> 双引擎。
          </div>
        )}
        {images.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
            {images.map((img, i) => (
              <div key={i} style={{ border: '1px solid var(--border-color)', borderRadius: 6, overflow: 'hidden', background: 'var(--bg-sidebar)' }}>
                <div style={{ position: 'relative', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 200 }}>
                  <img src={img.url} alt={img.prompt} style={{ maxWidth: '100%', maxHeight: 400, display: 'block' }} />
                </div>
                <div style={{ padding: '8px 12px', fontSize: 11 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <span style={{ padding: '1px 6px', borderRadius: 3, background: img.provider === 'minimax' ? '#FF6900' : '#1664FF', color: '#fff', fontSize: 10 }}>
                      {img.provider === 'minimax' ? 'MiniMax' : 'Volcengine'}
                    </span>
                    <div style={{ flex: 1 }} />
                    <button onClick={() => window.sky.app.openExternal(img.url)} title="在新窗口打开" style={{ padding: '2px 4px', background: 'transparent', border: 'none' }}>
                      <ExternalLink size={13} />
                    </button>
                    <a href={img.url} download title="下载" style={{ padding: '2px 4px', background: 'transparent', border: 'none', display: 'inline-flex', color: 'var(--fg-secondary)' }}>
                      <Download size={13} />
                    </a>
                  </div>
                  <div style={{ color: 'var(--fg-secondary)', whiteSpace: 'pre-wrap', maxHeight: 60, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {img.prompt}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
