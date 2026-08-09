import React, { useState } from 'react'
import { BrainCircuit, ChevronDown, ChevronRight, Loader2, Send } from 'lucide-react'

interface Step {
  id: number
  title: string
  content: string
  collapsed: boolean
}

export default function ReasoningView() {
  const [question, setQuestion] = useState('')
  const [maxSteps, setMaxSteps] = useState(8)
  const [loading, setLoading] = useState(false)
  const [plan, setPlan] = useState<string | null>(null)
  const [steps, setSteps] = useState<Step[]>([])
  const [answer, setAnswer] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const toggleStep = (idx: number) => {
    setSteps((prev) => prev.map((s, i) => (i === idx ? { ...s, collapsed: !s.collapsed } : s)))
  }

  const runReasoning = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    setError(null)
    setPlan(null)
    setSteps([])
    setAnswer(null)
    try {
      const res = await window.sky.api.post('/reasoning/deep', {
        question: question.trim(),
        max_steps: maxSteps
      })
      if (res.ok && res.data) {
        const d = res.data
        setPlan(d.plan || d.reasoning_plan || null)
        const rawSteps = d.steps || d.reasoning_steps || []
        setSteps(
          rawSteps.map((s: any, i: number) => ({
            id: i,
            title: s.title || s.step || `Step ${i + 1}`,
            content: typeof s === 'string' ? s : (s.content || s.thought || s.description || JSON.stringify(s)),
            collapsed: false
          }))
        )
        setAnswer(d.answer || d.final_answer || d.conclusion || null)
      } else {
        throw new Error(res.error || `HTTP ${res.status}`)
      }
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: 16, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--fg-assistant)', fontWeight: 600 }}>
            <BrainCircuit size={20} /> Deep Thinking
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
            <span style={{ color: 'var(--fg-secondary)' }}>Max steps: {maxSteps}</span>
            <input
              type="range"
              min={2}
              max={20}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value))}
              style={{ width: 120 }}
            />
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runReasoning() }
            }}
            placeholder="Ask a complex question that requires multi-step reasoning..."
            rows={2}
            style={{ flex: 1, minHeight: 52, padding: '10px 12px' }}
          />
          <button
            onClick={runReasoning}
            disabled={loading || !question.trim()}
            style={{
              height: 52, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff',
              display: 'flex', alignItems: 'center', gap: 6
            }}
          >
            {loading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Send size={16} />}
            Think
          </button>
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {error && (
          <div style={{
            padding: 12, borderRadius: 4, background: 'rgba(244,135,113,0.1)',
            border: '1px solid var(--error)', color: 'var(--error)', marginBottom: 16
          }}>
            Error: {error}
          </div>
        )}
        {loading && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--fg-secondary)' }}>
            <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} />
            Reasoning in progress, please wait...
          </div>
        )}
        {!loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 920 }}>
            {plan && (
              <div style={{ border: '1px solid var(--border-color)', borderRadius: 4 }}>
                <div style={{
                  padding: '10px 14px',
                  background: 'var(--bg-sidebar)',
                  borderBottom: '1px solid var(--border-color)',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8
                }}>
                  📋 Plan
                </div>
                <div style={{ padding: 14, whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{plan}</div>
              </div>
            )}
            {steps.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--fg-secondary)' }}>
                  REASONING STEPS ({steps.length})
                </div>
                {steps.map((s, i) => (
                  <div key={s.id} style={{ border: '1px solid var(--border-color)', borderRadius: 4, overflow: 'hidden' }}>
                    <div
                      onClick={() => toggleStep(i)}
                      style={{
                        padding: '10px 14px',
                        background: 'var(--bg-sidebar)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        userSelect: 'none'
                      }}
                    >
                      {s.collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                      <span style={{ color: 'var(--fg-secondary)', fontSize: 11 }}>Step {i + 1}</span>
                      <span style={{ fontWeight: 500 }}>{s.title}</span>
                    </div>
                    {!s.collapsed && (
                      <div style={{ padding: 14, whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: 13 }}>
                        {s.content}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            {answer && (
              <div style={{
                border: '1px solid var(--accent)', borderRadius: 4, overflow: 'hidden',
                background: 'rgba(0,122,204,0.08)'
              }}>
                <div style={{
                  padding: '10px 14px',
                  background: 'var(--accent)',
                  color: '#fff',
                  fontWeight: 600
                }}>
                  ✅ Final Answer
                </div>
                <div style={{ padding: 16, whiteSpace: 'pre-wrap', lineHeight: 1.65, fontSize: 14 }}>
                  {answer}
                </div>
              </div>
            )}
            {!plan && steps.length === 0 && !answer && !error && (
              <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60, lineHeight: 1.7 }}>
                Enter a complex question to start deep reasoning. <br />
                Example: &quot;How many prime numbers between 1 and 100 are divisible by the sum of their digits?&quot;
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
