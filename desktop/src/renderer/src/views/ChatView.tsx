import React, { useState, useRef, useEffect } from 'react'
import { Send, User, Bot, ChevronDown, Server, Cpu, CheckCircle, Loader2 } from 'lucide-react'
import type { ProviderMeta, ProviderConfig, ProvidersStore } from '../../preload/index'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  streaming?: boolean
}

type Engine = 'auto' | 'local-backend' | 'provider'

export default function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'system',
      content: '欢迎使用 Sky V1 Omni Chat！你可以在下方与模型对话。\n\n💡 推荐流程：先打开侧边栏「设置 → 模型服务商」填写火山/小米等厂商的 API Key，测通后选「External Provider」引擎即可体验 SSE 流式打字效果。若服务商未配置，默认回退到本地 Python 后端 (/v1/chat/completions)。',
      timestamp: Date.now()
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [engine, setEngine] = useState<Engine>('auto')
  const [selectedProvider, setSelectedProvider] = useState<string>('volcengine')
  const [providers, setProviders] = useState<ProviderMeta[]>([])
  const [store, setStore] = useState<ProvidersStore | null>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const abortCtrlRef = useRef<AbortController | null>(null)

  const loadProviders = async () => {
    const [m, s] = await Promise.all([window.sky.providers.listMeta(), window.sky.providers.getStore()])
    setProviders(m)
    setStore(s)
    setSelectedProvider(s.activeProvider || s.providers.volcengine?.id || m[0]?.id || 'volcengine')
    if (s.providers[s.activeProvider]?.apiKey && s.providers[s.activeProvider]?.model) {
      setEngine('auto')
    }
  }
  useEffect(() => {
    loadProviders()
  }, [])

  useEffect(() => {
    const off = window.sky.providers.onChatDelta(({ delta }) => {
      setMessages((prev) => {
        if (prev.length === 0) return prev
        const last = prev[prev.length - 1]
        if (!last.streaming) return prev
        return [...prev.slice(0, -1), { ...last, content: last.content + delta }]
      })
    })
    return () => { off() }
  }, [])

  useEffect(() => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight
    }
  }, [messages, loading])

  const effectiveProvider = providers.find((p) => p.id === selectedProvider)
  const effectiveCfg: ProviderConfig | undefined = store?.providers?.[selectedProvider]

  const providerReady = !!(effectiveCfg?.apiKey && effectiveCfg?.model)

  const shouldUseProvider = engine === 'provider' || (engine === 'auto' && providerReady)

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now()
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const historyForApi = messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m) => ({ role: m.role, content: m.content }))
      const allMessages = [...historyForApi, { role: 'user', content: text }]

      const assistantId = `a-${Date.now()}`

      if (shouldUseProvider) {
        // ===== 经 providers 走外部 API（SSE 流式，火山/小米/通义/智谱等）=====
        if (!providerReady) throw new Error(`当前服务商「${effectiveProvider?.name}」未配置完整，请先到「设置 → 模型服务商」填写 API Key 和模型/接入点ID。`)
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: '', timestamp: Date.now(), streaming: true }
        ])
        const result = await window.sky.providers.chat({
          providerId: selectedProvider,
          messages: allMessages,
          max_tokens: effectiveCfg?.maxTokens || 2048,
          temperature: effectiveCfg?.temperature ?? 0.7,
          stream: true
        })
        // 流式推送已经通过 onChatDelta 增量叠加到 message，这里收尾补全 content
        if (result.ok) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: result.content, streaming: false } : m
            )
          )
        } else {
          throw new Error(result.error || `请求失败 HTTP ${result.status}`)
        }
      } else {
        // ===== 走本地 Python 后端 (/v1/chat/completions) =====
        const result = await window.sky.api.post('/chat/completions', {
          messages: allMessages,
          max_tokens: effectiveCfg?.maxTokens || 512,
          temperature: effectiveCfg?.temperature ?? 0.7
        })
        if (result.ok && result.data) {
          const content =
            typeof result.data === 'string'
              ? result.data
              : result.data.choices?.[0]?.message?.content ||
                result.data.choices?.[0]?.text ||
                JSON.stringify(result.data, null, 2)
          setMessages((prev) => [
            ...prev,
            { id: assistantId, role: 'assistant', content, timestamp: Date.now() }
          ])
        } else {
          throw new Error(result.error || `HTTP ${result.status}`)
        }
      }
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content: `❌ Error: ${e.message || String(e)}\n\n${shouldUseProvider ? '提示：点击上方「服务商」可切换到已配置的厂商；或去设置页测试连通性。' : '提示：请确保本地 Python 后端运行中，或在设置页配置外部服务商 API Key。'}`,
          timestamp: Date.now()
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 引擎选择 Toolbar */}
      <div style={{
        padding: '8px 16px',
        borderBottom: '1px solid var(--border-color)',
        background: 'var(--bg-sidebar)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        fontSize: 12,
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 8px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
          <span style={{ color: 'var(--fg-secondary)' }}>引擎：</span>
          <select
            value={engine}
            onChange={(e) => setEngine(e.target.value as Engine)}
            style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12, padding: '4px 2px' }}
          >
            <option value="auto">Auto（优先外部，否则本地后端）</option>
            <option value="provider">External Provider（火山/小米等 SSE 流式）</option>
            <option value="local-backend">Local Python Backend（/v1/chat）</option>
          </select>
          <ChevronDown size={12} />
        </div>

        {(engine === 'provider' || (engine === 'auto' && providerReady)) && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 8px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
              <Server size={13} style={{ color: effectiveProvider?.brandColor }} />
              <span style={{ color: 'var(--fg-secondary)' }}>服务商：</span>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12, padding: '4px 2px' }}
              >
                {providers.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
              <ChevronDown size={12} />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {providerReady ? (
                <span style={{ color: 'var(--success)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <CheckCircle size={12} /> 已就绪（SSE 流式）
                </span>
              ) : (
                <span style={{ color: 'var(--warning)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <Loader2 size={12} style={{ animation: 'spin 0.8s linear infinite' }} /> 需填写 API Key + Model
                </span>
              )}
            </div>
          </>
        )}

        {engine === 'local-backend' && (
          <span style={{ color: 'var(--fg-secondary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <Cpu size={12} /> 调用本地 Python 后端（需在运行中）
          </span>
        )}

        <div style={{ flex: 1 }} />

        <div style={{ color: 'var(--fg-secondary)', fontSize: 11 }}>
          当前：{shouldUseProvider ? (effectiveProvider?.name || '外部服务商') : '本地 Python 后端'}
        </div>
      </div>

      <div
        ref={scrollerRef}
        style={{ flex: 1, overflow: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 14 }}
      >
        {messages.map((m) => (
          <div
            key={m.id}
            style={{
              display: 'flex',
              gap: 10,
              maxWidth: '85%',
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start'
            }}
          >
            {m.role !== 'user' && (
              <div style={{
                width: 28, height: 28, borderRadius: 4, flexShrink: 0,
                background: m.role === 'system' ? '#444' : 'var(--fg-assistant)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff'
              }}>
                {m.role === 'system' ? '!' : <Bot size={16} />}
              </div>
            )}
            <div style={{
              background: m.role === 'user' ? '#094771' : '#2d2d30',
              padding: '10px 14px',
              borderRadius: 6,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              fontSize: 13,
              lineHeight: 1.55,
              border: '1px solid var(--border-color)',
              position: 'relative'
            }}>
              {m.content || (m.streaming ? <span style={{ opacity: 0.4 }}>▊</span> : '')}
              {m.streaming && (
                <span style={{
                  display: 'inline-block',
                  width: 7, height: 15,
                  background: 'var(--accent)',
                  marginLeft: 2,
                  verticalAlign: '-2px',
                  animation: 'blink 1s steps(1) infinite'
                }} />
              )}
            </div>
            {m.role === 'user' && (
              <div style={{
                width: 28, height: 28, borderRadius: 4, flexShrink: 0,
                background: 'var(--fg-user)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#111'
              }}>
                <User size={16} />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', gap: 10, alignSelf: 'flex-start' }}>
            <div style={{
              width: 28, height: 28, borderRadius: 4, background: 'var(--fg-assistant)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff'
            }}>
              <Bot size={16} />
            </div>
            <div style={{ background: '#2d2d30', padding: '10px 14px', borderRadius: 6, color: 'var(--fg-secondary)' }}>
              {shouldUseProvider ? 'Generating via SSE...' : 'Calling backend...'}
            </div>
          </div>
        )}
      </div>
      <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
            rows={2}
            style={{ flex: 1, minHeight: 44, fontFamily: 'inherit', padding: '10px 12px' }}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            style={{
              height: 44,
              padding: '0 16px',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: loading ? 'var(--bg-input)' : 'var(--accent)',
              color: '#fff',
              border: 'none'
            }}
          >
            <Send size={16} /> Send
          </button>
        </div>
      </div>
    </div>
  )
}
