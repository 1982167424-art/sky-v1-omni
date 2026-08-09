import React, { useState, useEffect } from 'react'
import { Settings, RefreshCw, Power, Download, Info, Server, Eye, EyeOff, Save, Trash2, CheckCircle, XCircle, Loader2, ExternalLink, Wand2 } from 'lucide-react'
import { useApp } from '../store'
import type { ProviderMeta, ProviderConfig, ProvidersStore } from '../../preload/index'

type Tab = 'app' | 'backend' | 'providers' | 'about'

export default function SettingsView() {
  const { state, dispatch } = useApp()
  const [tab, setTab] = useState<Tab>('providers')
  const [checkingUpdate, setCheckingUpdate] = useState(false)
  const [updateStatus, setUpdateStatus] = useState<string>('')

  const restart = async () => {
    await window.sky.backend.restart()
  }
  const stop = async () => {
    await window.sky.backend.stop()
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        {([
          { id: 'providers', label: '模型服务商', icon: Server },
          { id: 'app', label: 'Application', icon: Settings },
          { id: 'backend', label: '后端服务', icon: Power },
          { id: 'about', label: 'About', icon: Info }
        ] as { id: Tab; label: string; icon: any }[]).map(({ id, label, icon: Icon }) => (
          <div
            key={id}
            onClick={() => setTab(id)}
            style={{
              padding: '0 18px',
              height: 44,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              cursor: 'pointer',
              fontSize: 13,
              borderBottom: tab === id ? '2px solid var(--accent)' : '2px solid transparent',
              color: tab === id ? 'var(--fg-primary)' : 'var(--fg-secondary)',
              marginTop: tab === id ? 0 : 2
            }}
          >
            <Icon size={16} /> {label}
          </div>
        ))}
      </div>
      <div style={{ padding: 24, overflow: 'auto', flex: 1 }}>
        <div style={{ maxWidth: 960, margin: '0 auto' }}>
          {tab === 'providers' && <ProvidersPanel />}
          {tab === 'app' && (
            <Section title="Application">
              <Row label="Version">
                <span style={{ color: 'var(--fg-secondary)' }}>v{state.version}</span>
              </Row>
              <Row label="App Updates">
                <button
                  onClick={async () => {
                    setCheckingUpdate(true)
                    setUpdateStatus('Checking for updates...')
                    try {
                      const notes = await window.sky.github.getReleaseNotes()
                      setUpdateStatus(notes ? `Latest: ${notes.tag_name} (${notes.name})` : 'Already on latest or release info unavailable')
                    } catch {
                      setUpdateStatus('Update check failed (requires gh CLI)')
                    } finally { setCheckingUpdate(false) }
                  }}
                  disabled={checkingUpdate}
                  style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                >
                  {checkingUpdate ? <RefreshCw size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Download size={14} />}
                  Check for updates
                </button>
              </Row>
              {updateStatus && (
                <div style={{ padding: '10px 14px', background: 'var(--bg-sidebar)', border: '1px solid var(--border-color)', borderRadius: 4, fontSize: 12, color: 'var(--fg-secondary)', marginTop: -4 }}>
                  {updateStatus}
                </div>
              )}
            </Section>
          )}
          {tab === 'backend' && (
            <>
              <Section title="Backend (Python sky-v1-omni)">
                <Row label="Status">
                  <span style={{
                    padding: '2px 8px', borderRadius: 3, fontSize: 12,
                    background: state.backendStatus.running ? 'rgba(78,201,176,0.15)' : 'rgba(244,135,113,0.15)',
                    color: state.backendStatus.running ? 'var(--success)' : 'var(--error)'
                  }}>
                    {state.backendStatus.running ? '● Running' : '● Stopped'} on port {state.backendStatus.port}
                  </span>
                </Row>
                <Row label="Controls">
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button onClick={restart} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <RefreshCw size={14} /> Restart
                    </button>
                    <button onClick={stop} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Power size={14} /> Stop
                    </button>
                  </div>
                </Row>
                <Row label="Panel Terminal">
                  <button onClick={() => dispatch({ type: 'TOGGLE_TERMINAL' })}>
                    {state.terminalOpen ? 'Hide' : 'Show'} Panel
                  </button>
                </Row>
              </Section>
              <Section title="Recent Backend Logs (tail 50)">
                <div style={{
                  background: '#111', border: '1px solid var(--border-color)',
                  borderRadius: 4, padding: 12, fontFamily: 'Consolas, Monaco, monospace',
                  fontSize: 11, maxHeight: 280, overflow: 'auto', color: '#cccccc',
                  whiteSpace: 'pre-wrap', lineHeight: 1.5
                }}>
                  {state.logs.length === 0 ? (
                    <span style={{ color: 'var(--fg-secondary)' }}>(no logs yet)</span>
                  ) : state.logs.slice(-50).join('\n')}
                </div>
              </Section>
            </>
          )}
          {tab === 'about' && (
            <Section title="About">
              <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', padding: '10px 4px' }}>
                <Info size={16} color="var(--fg-secondary)" />
                <div style={{ fontSize: 12, color: 'var(--fg-secondary)', lineHeight: 1.7 }}>
                  <div style={{ color: 'var(--fg-primary)', fontWeight: 600, marginBottom: 4 }}>Sky V1 Omni Desktop</div>
                  Electron + React desktop wrapper for the sky-v1-omni backend.<br />
                  内置多厂商模型接入，支持火山方舟 SSE 流式推理、小米、通义千问、智谱、月之暗面、DeepSeek、OpenAI 兼容。<br />
                  GitHub 集成需要 `gh` CLI。
                </div>
              </div>
            </Section>
          )}
        </div>
      </div>
    </div>
  )
}

// ============================================================
// 模型服务商面板（重头戏）
// ============================================================

function ProvidersPanel() {
  const [meta, setMeta] = useState<ProviderMeta[]>([])
  const [store, setStore] = useState<ProvidersStore | null>(null)
  const [selected, setSelected] = useState<string>('volcengine')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<Record<string, { ok: boolean; status: number; latencyMs: number; sample?: string; error?: string } | null>>({})
  const [savedFlash, setSavedFlash] = useState<string | null>(null)
  const [showKey, setShowKey] = useState<Record<string, boolean>>({})
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  // 本地 draft 状态：允许自由编辑，失焦时才保存到主进程（H2 修复）
  const [draft, setDraft] = useState<Record<string, Partial<ProviderConfig>>>({})

  const load = async () => {
    setLoading(true)
    setErrorMsg(null)
    try {
      const [m, s] = await Promise.all([window.sky.providers.listMeta(), window.sky.providers.getStore()])
      setMeta(m)
      setStore(s)
      setSelected(s.activeProvider || m[0]?.id || 'volcengine')
    } catch (e: any) {
      setErrorMsg(e.message || String(e))
    } finally { setLoading(false) }
  }
  useEffect(() => { load() }, [])

  const currentMeta = meta.find((m) => m.id === selected)
  const currentCfg: ProviderConfig | undefined = store?.providers?.[selected]
  const draftCfg = draft[selected]
  // 优先用 draft（用户正在编辑的值），否则用 store（可能带掩码）
  const apiKeyVal = draftCfg?.apiKey ?? currentCfg?.apiKey ?? ''
  const modelVal = draftCfg?.model ?? currentCfg?.model ?? ''
  const baseUrlVal = draftCfg?.baseUrl ?? currentCfg?.baseUrl ?? currentMeta?.defaultBaseUrl ?? ''
  const tempVal = draftCfg?.temperature ?? currentCfg?.temperature ?? 0.7
  const maxTokensVal = draftCfg?.maxTokens ?? currentCfg?.maxTokens ?? 2048

  const updateDraft = (field: keyof ProviderConfig, value: any) => {
    setDraft((prev) => ({ ...prev, [selected]: { ...prev[selected], [field]: value } }))
  }

  const saveField = async (field: keyof ProviderConfig, value: any) => {
    if (!currentCfg) return
    setSaving(true)
    setErrorMsg(null)
    try {
      const r = await window.sky.providers.update(selected, { [field]: value, enabled: true } as any)
      if (r.ok && r.store) {
        setStore(r.store)
        setSavedFlash(`✓ 已保存「${currentMeta?.name}」配置`)
        setTimeout(() => setSavedFlash(null), 1500)
      } else {
        setErrorMsg(r.error || '保存失败')
      }
    } catch (e: any) {
      setErrorMsg(e.message || String(e))
    } finally { setSaving(false) }
  }

  const activate = async () => {
    setErrorMsg(null)
    try {
      const r = await window.sky.providers.setActive(selected)
      if (r.ok && r.store) setStore(r.store)
      else setErrorMsg(r.error || '设为默认失败')
    } catch (e: any) {
      setErrorMsg(e.message || String(e))
    }
  }

  const removeKey = async () => {
    setErrorMsg(null)
    try {
      const r = await window.sky.providers.removeKey(selected)
      if (r.ok && r.store) {
        setStore(r.store)
        setDraft((prev) => ({ ...prev, [selected]: { ...prev[selected], apiKey: '' } }))
      } else setErrorMsg(r.error || '清除失败')
    } catch (e: any) {
      setErrorMsg(e.message || String(e))
    }
  }

  const test = async () => {
    setTesting(selected)
    setTestResult((s) => ({ ...s, [selected]: null }))
    setErrorMsg(null)
    try {
      const r = await window.sky.providers.testConnection(selected)
      setTestResult((s) => ({ ...s, [selected]: r as any }))
    } catch (e: any) {
      setErrorMsg(e.message || String(e))
    } finally {
      setTesting(null)
    }
  }

  if (loading) {
    return <div style={{ color: 'var(--fg-secondary)' }}><Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite', display: 'inline', marginRight: 8 }} />加载服务商配置…</div>
  }

  return (
    <div style={{ display: 'flex', gap: 16, minHeight: 480 }}>
      {/* 左侧：服务商列表 */}
      <div style={{ width: 280, flexShrink: 0, border: '1px solid var(--border-color)', borderRadius: 6, background: 'var(--bg-sidebar)', overflow: 'hidden' }}>
        <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border-color)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--fg-secondary)', fontWeight: 700 }}>
          支持的服务商
        </div>
        <div>
          {meta.map((m) => {
            const cfg = store?.providers?.[m.id]
            const active = store?.activeProvider === m.id
            const configured = !!(cfg && cfg.apiKey && cfg.model)
            const sel = selected === m.id
            return (
              <div
                key={m.id}
                onClick={() => { setSelected(m.id); setDraft({}) }}
                style={{
                  padding: '10px 14px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  cursor: 'pointer',
                  background: sel ? 'rgba(255,255,255,0.06)' : 'transparent',
                  borderLeft: sel ? `3px solid ${m.brandColor}` : '3px solid transparent'
                }}
              >
                <div style={{
                  width: 26, height: 26, borderRadius: 6, flexShrink: 0,
                  background: m.brandColor, color: '#fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 800
                }}>
                  {m.name.charAt(0)}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, color: 'var(--fg-primary)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: 6 }}>
                    {m.name}
                    {active && <span style={{ fontSize: 10, background: 'var(--accent)', color: '#fff', padding: '0 5px', borderRadius: 3 }}>使用中</span>}
                  </div>
                  <div style={{ fontSize: 11, color: configured ? 'var(--success)' : 'var(--fg-secondary)', marginTop: 2 }}>
                    {configured ? (m.supportsStreaming ? '✓ 已配置 · 支持流式' : '✓ 已配置') : '未配置 API Key / Model'}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 右侧：当前服务商表单 */}
      <div style={{ flex: 1, border: '1px solid var(--border-color)', borderRadius: 6, background: 'var(--bg-editor)' }}>
        {!currentMeta || !currentCfg ? null : (
          <>
            <div style={{
              padding: '14px 18px',
              borderBottom: '1px solid var(--border-color)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              background: `linear-gradient(180deg, ${currentMeta.brandColor}11, transparent)`
            }}>
              <div style={{
                width: 38, height: 38, borderRadius: 8,
                background: currentMeta.brandColor, color: '#fff',
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, fontWeight: 800
              }}>
                {currentMeta.name.charAt(0)}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 15, fontWeight: 600 }}>{currentMeta.name}</div>
                <div style={{ fontSize: 11, color: 'var(--fg-secondary)', marginTop: 2 }}>
                  Base URL 默认：<code style={{ fontSize: 11, background: 'var(--bg-input)', padding: '1px 6px', borderRadius: 3 }}>{currentMeta.defaultBaseUrl}</code>
                  {currentMeta.id === 'volcengine' && ' · ⚠️ model 字段必须填【推理接入点 Endpoint ID】(ep-xxxx)，不是模型名'}
                </div>
              </div>
              <a
                onClick={(e) => { e.preventDefault(); window.sky.app.openExternal(currentMeta.docsUrl) }}
                href="#"
                style={{ fontSize: 12, color: 'var(--accent)', display: 'inline-flex', alignItems: 'center', gap: 4 }}
              >
                官方文档 <ExternalLink size={12} />
              </a>
            </div>

            <div style={{ padding: 18 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px 18px' }}>
                <Field label="API Key" required extra={
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button onClick={() => setShowKey((s) => ({ ...s, [currentMeta.id]: !s[currentMeta.id] }))} title="显示/隐藏" style={{ padding: '2px 8px' }}>
                      {showKey[currentMeta.id] ? <EyeOff size={13} /> : <Eye size={13} />}
                    </button>
                    {currentCfg.apiKey && (
                      <button onClick={removeKey} title="清空 Key" style={{ padding: '2px 8px' }}>
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                }>
                  <input
                    type={showKey[currentMeta.id] ? 'text' : 'password'}
                    value={apiKeyVal}
                    onChange={(e) => updateDraft('apiKey', e.target.value)}
                    onBlur={(e) => saveField('apiKey', e.target.value)}
                    placeholder={`请粘贴 ${currentMeta.name} 的 API Key…`}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: 11, color: 'var(--fg-secondary)', marginTop: 4 }}>
                    密钥仅保存在本机 userData/providers.json（0600 权限），绝不打印到日志。输入完毕失焦后自动保存。
                  </div>
                </Field>

                <Field label={currentMeta.modelLabel} required>
                  <input
                    type="text"
                    value={modelVal}
                    onChange={(e) => updateDraft('model', e.target.value)}
                    onBlur={(e) => saveField('model', e.target.value)}
                    placeholder={currentMeta.modelPlaceholder}
                    style={{ width: '100%' }}
                  />
                  {currentMeta.id === 'volcengine' && (
                    <div style={{ fontSize: 11, color: 'var(--warning)', marginTop: 4 }}>
                      ⚠️ 火山方舟用「接入点ID」而不是模型名，例如 <code>ep-2025xxxxxx-xxxxx</code>，到火山控制台 / 模型推理接入点复制。
                    </div>
                  )}
                </Field>

                <Field label="Base URL（可覆盖）">
                  <input
                    type="text"
                    value={baseUrlVal}
                    onChange={(e) => updateDraft('baseUrl', e.target.value)}
                    onBlur={(e) => saveField('baseUrl', e.target.value)}
                    placeholder={currentMeta.defaultBaseUrl}
                    style={{ width: '100%' }}
                  />
                </Field>

                <Field label="Temperature (0~2)">
                  <input
                    type="number"
                    min={0} max={2} step={0.1}
                    value={tempVal}
                    onChange={(e) => updateDraft('temperature', parseFloat(e.target.value) || 0)}
                    onBlur={(e) => saveField('temperature', parseFloat(e.target.value) || 0)}
                    style={{ width: '100%' }}
                  />
                </Field>

                <Field label="Max Output Tokens">
                  <input
                    type="number"
                    min={1} max={32768} step={1}
                    value={maxTokensVal}
                    onChange={(e) => updateDraft('maxTokens', Math.max(1, parseInt(e.target.value) || 1))}
                    onBlur={(e) => saveField('maxTokens', Math.max(1, parseInt(e.target.value) || 1))}
                    style={{ width: '100%' }}
                  />
                </Field>

                <Field label="启用该服务商并设为默认">
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button
                      onClick={activate}
                      disabled={store?.activeProvider === currentMeta.id}
                      style={{
                        background: store?.activeProvider === currentMeta.id ? 'var(--success)' : 'var(--accent)',
                        color: '#fff',
                        border: 'none',
                        padding: '6px 14px',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6
                      }}
                    >
                      <Wand2 size={14} />
                      {store?.activeProvider === currentMeta.id ? '✓ 当前默认' : '设为默认'}
                    </button>
                    <div style={{ fontSize: 11, color: 'var(--fg-secondary)' }}>
                      聊天、推理等面板默认优先调用「默认服务商」
                    </div>
                  </div>
                </Field>
              </div>

              {/* 操作栏：测试 / 保存 / 状态 */}
              <div style={{
                marginTop: 22,
                padding: 14,
                border: '1px solid var(--border-color)',
                borderRadius: 6,
                background: 'var(--bg-sidebar)',
                display: 'flex',
                gap: 12,
                alignItems: 'center',
                flexWrap: 'wrap'
              }}>
                <button
                  onClick={async () => {
                    setSaving(true)
                    setErrorMsg(null)
                    try {
                      const r = await window.sky.providers.update(currentMeta.id, { enabled: true })
                      if (r.ok && r.store) setStore(r.store)
                      else setErrorMsg(r.error || '保存失败')
                    } catch (e: any) {
                      setErrorMsg(e.message || String(e))
                    } finally { setSaving(false) }
                  }}
                  disabled={saving}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 14px' }}
                >
                  <Save size={14} /> {saving ? '保存中…' : '保存配置'}
                </button>
                <button
                  onClick={test}
                  disabled={testing === currentMeta.id}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 14px' }}
                >
                  {testing === currentMeta.id
                    ? <Loader2 size={14} style={{ animation: 'spin 0.8s linear infinite' }} />
                    : <CheckCircle size={14} />}
                  测试连通性
                </button>
                <div style={{ flex: 1 }} />
                {savedFlash && (
                  <div style={{ color: 'var(--success)', fontSize: 12 }}>{savedFlash}</div>
                )}
              </div>

              {/* 错误提示 */}
              {errorMsg && (
                <div style={{ marginTop: 14, padding: 10, borderRadius: 4, border: '1px solid var(--error)', background: 'rgba(244,135,113,0.1)', color: 'var(--error)', fontSize: 12 }}>
                  {errorMsg}
                </div>
              )}

              {/* 测试结果 */}
              {testResult[currentMeta.id] && (
                <div style={{
                  marginTop: 14,
                  padding: 14,
                  borderRadius: 6,
                  border: `1px solid ${testResult[currentMeta.id]!.ok ? 'var(--success)' : 'var(--error)'}`,
                  background: testResult[currentMeta.id]!.ok ? 'rgba(78,201,176,0.08)' : 'rgba(244,135,113,0.08)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    {testResult[currentMeta.id]!.ok
                      ? <CheckCircle size={16} color="var(--success)" />
                      : <XCircle size={16} color="var(--error)" />}
                    <span style={{ fontWeight: 600 }}>
                      {testResult[currentMeta.id]!.ok ? '连接成功 ✓' : '连接失败'}
                      <span style={{ marginLeft: 10, color: 'var(--fg-secondary)', fontWeight: 400, fontSize: 12 }}>
                        HTTP {testResult[currentMeta.id]!.status} · {testResult[currentMeta.id]!.latencyMs}ms
                      </span>
                    </span>
                  </div>
                  {testResult[currentMeta.id]!.sample && (
                    <div style={{ fontSize: 12, color: 'var(--fg-secondary)', marginBottom: 6 }}>
                      <span style={{ color: 'var(--fg-primary)' }}>模型回复预览：</span>
                      <span style={{ fontFamily: 'Consolas, Monaco, monospace', background: 'var(--bg-input)', padding: '2px 6px', borderRadius: 3, marginLeft: 6 }}>
                        {JSON.stringify(testResult[currentMeta.id]!.sample)}
                      </span>
                    </div>
                  )}
                  {testResult[currentMeta.id]!.error && (
                    <div style={{ fontSize: 12, color: 'var(--error)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                      错误详情：{testResult[currentMeta.id]!.error}
                    </div>
                  )}
                </div>
              )}

              <div style={{ marginTop: 20, fontSize: 11, color: 'var(--fg-secondary)', lineHeight: 1.7 }}>
                <div style={{ fontWeight: 600, color: 'var(--fg-primary)', marginBottom: 4 }}>💡 配置流程（建议按此顺序）</div>
                1) 从左栏选择服务商 → 2) 填入 API Key + 模型/接入点 ID → 3) 点击「保存配置」→ 4) 点击「测试连通性」→ 5) 成功后点击「设为默认」
                <br />
                配置保存后立即在 Chat 面板可用（Chat 会走 SSE 流式输出，体验实时打字效果）。
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function Field({ label, required, extra, children }: { label: string; required?: boolean; extra?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--fg-primary)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>{label}{required && <span style={{ color: 'var(--error)' }}> *</span>}</span>
        <div style={{ flex: 1 }} />
        {extra}
      </div>
      {children}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--fg-secondary)', paddingBottom: 8, borderBottom: '1px solid var(--border-color)', marginBottom: 12 }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {children}
      </div>
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', padding: '8px 4px', gap: 16, minHeight: 40 }}>
      <div style={{ width: 160, color: 'var(--fg-primary)', fontSize: 13, flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  )
}
