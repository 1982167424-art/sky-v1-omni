import React, { useState } from 'react'
import { Settings, RefreshCw, Power, Download, Info } from 'lucide-react'
import { useApp } from '../store'

export default function SettingsView() {
  const { state, dispatch } = useApp()
  const [checkingUpdate, setCheckingUpdate] = useState(false)
  const [updateStatus, setUpdateStatus] = useState<string>('')

  const restart = async () => {
    await window.sky.backend.restart()
  }
  const stop = async () => {
    await window.sky.backend.stop()
  }

  return (
    <div style={{ padding: 24, overflow: 'auto', height: '100%' }}>
      <div style={{ maxWidth: 720 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
          <Settings size={22} color="var(--accent)" />
          <h2 style={{ fontSize: 18, fontWeight: 600 }}>Settings</h2>
        </div>

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

        <Section title="Backend (Python)">
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
            <button
              onClick={() => dispatch({ type: 'TOGGLE_TERMINAL' })}
            >
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

        <Section title="About">
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', padding: '10px 4px' }}>
            <Info size={16} color="var(--fg-secondary)" />
            <div style={{ fontSize: 12, color: 'var(--fg-secondary)', lineHeight: 1.7 }}>
              <div style={{ color: 'var(--fg-primary)', fontWeight: 600, marginBottom: 4 }}>Sky V1 Omni Desktop</div>
              An Electron + React desktop wrapper for the sky-v1-omni backend.<br />
              Requires Python 3.10+ with the sky_v1 package available in PYTHONPATH (repo root).<br />
              GitHub features require the `gh` CLI installed and authenticated.
            </div>
          </div>
        </Section>
      </div>
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
