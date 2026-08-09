import React, { useState, useRef, useEffect } from 'react'
import { Send, User, Bot } from 'lucide-react'
import clsx from 'clsx'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
}

export default function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'system',
      content: 'Welcome to Sky V1 Omni Chat. Ask me anything!',
      timestamp: Date.now()
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight
    }
  }, [messages])

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
      const result = await window.sky.api.post('/chat/completions', {
        messages: allMessages,
        max_tokens: 512,
        temperature: 0.7
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
          {
            id: `a-${Date.now()}`,
            role: 'assistant',
            content,
            timestamp: Date.now()
          }
        ])
      } else {
        throw new Error(result.error || `HTTP ${result.status}`)
      }
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${e.message || String(e)}. Is the backend running?`,
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
              border: '1px solid var(--border-color)'
            }}>
              {m.content}
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
              Thinking...
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
