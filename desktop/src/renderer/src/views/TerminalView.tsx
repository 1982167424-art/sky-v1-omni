import React, { useEffect, useRef, useState } from 'react'
import { Terminal as TerminalIcon, Play, Square, Power } from 'lucide-react'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'
import { WebLinksAddon } from 'xterm-addon-web-links'
import 'xterm/css/xterm.css'

interface Props {
  embedded?: boolean
}

export default function TerminalView({ embedded }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitRef = useRef<FitAddon | null>(null)
  const [spawned, setSpawned] = useState(false)
  const [spawnError, setSpawnError] = useState<string | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const term = new Terminal({
      fontFamily: 'Consolas, "Courier New", monospace',
      fontSize: 13,
      lineHeight: 1.2,
      theme: {
        background: '#1e1e1e',
        foreground: '#cccccc',
        cursor: '#cccccc',
        cursorAccent: '#1e1e1e',
        black: '#000000',
        red: '#cd3131',
        green: '#0dbc79',
        yellow: '#e5e510',
        blue: '#2472c8',
        magenta: '#bc3fbc',
        cyan: '#11a8cd',
        white: '#e5e5e5',
        brightBlack: '#666666',
        brightRed: '#f14c4c',
        brightGreen: '#23d18b',
        brightYellow: '#f5f543',
        brightBlue: '#3b8eea',
        brightMagenta: '#d670d6',
        brightCyan: '#29b8db',
        brightWhite: '#ffffff'
      },
      cursorBlink: true,
      convertEol: true,
      scrollback: 5000
    })
    const fitAddon = new FitAddon()
    const linksAddon = new WebLinksAddon((_e, uri) => window.sky.app.openExternal(uri))
    term.loadAddon(fitAddon)
    term.loadAddon(linksAddon)
    term.open(containerRef.current)
    try { fitAddon.fit() } catch {}
    term.writeln('\x1b[33mSky V1 Omni Terminal\x1b[0m')
    term.writeln('Click "Start Shell" to begin, or type directly for local echo.\r\n')

    term.onData((data) => {
      if (spawned) {
        window.sky.terminal.input(data)
      } else {
        if (data === '\r') term.write('\r\n')
        else if (data === '\x7f') term.write('\b \b')
        else term.write(data)
      }
    })

    termRef.current = term
    fitRef.current = fitAddon

    const ro = new ResizeObserver(() => { try { fitAddon.fit() } catch {} })
    ro.observe(containerRef.current)

    const offData = window.sky.terminal.onData((data) => {
      term.write(data)
    })
    const offExit = window.sky.terminal.onExit(() => {
      setSpawned(false)
      term.writeln('\r\n\x1b[31m[Process exited]\x1b[0m\r\n')
    })

    return () => {
      offData?.()
      offExit?.()
      ro.disconnect()
      term.dispose()
    }
  }, [])

  useEffect(() => {
    const t = setTimeout(() => {
      try { fitRef.current?.fit() } catch {}
    }, 100)
    return () => clearTimeout(t)
  }, [embedded])

  const startShell = async () => {
    setSpawnError(null)
    const term = termRef.current
    const cols = term?.cols || 80
    const rows = term?.rows || 24
    const r = await window.sky.terminal.spawn(undefined, cols, rows)
    if (r.ok) {
      setSpawned(true)
    } else {
      setSpawnError(r.error || 'Failed to spawn')
    }
  }

  const killShell = async () => {
    await window.sky.terminal.kill()
    setSpawned(false)
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {!embedded && (
        <div style={{
          padding: 10, borderBottom: '1px solid var(--border-color)',
          background: 'var(--bg-sidebar)', display: 'flex', alignItems: 'center', gap: 10
        }}>
          <TerminalIcon size={16} color="var(--success)" />
          <span style={{ fontWeight: 600 }}>Terminal</span>
          <div style={{ flex: 1 }} />
          {!spawned ? (
            <button onClick={startShell} style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'var(--success)', border: 'none', color: '#111' }}>
              <Play size={14} /> Start Shell
            </button>
          ) : (
            <button onClick={killShell} style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'var(--error)', border: 'none', color: '#fff' }}>
              <Square size={12} fill="#fff" /> Kill
            </button>
          )}
          <div style={{ fontSize: 11, color: spawned ? 'var(--success)' : 'var(--fg-secondary)' }}>
            {spawned ? '● Running' : '○ Idle'}
          </div>
        </div>
      )}
      {spawnError && (
        <div style={{ padding: 8, fontSize: 12, color: 'var(--error)', background: 'rgba(244,135,113,0.1)', borderBottom: '1px solid var(--error)' }}>
          Failed to spawn shell: {spawnError}. Ensure node-pty compiled correctly.
        </div>
      )}
      {embedded && (
        <div style={{
          padding: '6px 10px', display: 'flex', alignItems: 'center', gap: 10,
          background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border-color)'
        }}>
          {!spawned ? (
            <button onClick={startShell} style={{ padding: '3px 10px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 5, background: 'var(--success)', border: 'none', color: '#111' }}>
              <Power size={12} /> Start Shell
            </button>
          ) : (
            <button onClick={killShell} style={{ padding: '3px 10px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 5, background: 'var(--error)', border: 'none', color: '#fff' }}>
              <Square size={10} fill="#fff" /> Kill
            </button>
          )}
          <div style={{ fontSize: 11, color: spawned ? 'var(--success)' : 'var(--fg-secondary)' }}>
            {spawned ? '● shell running' : '○ shell not started'}
          </div>
        </div>
      )}
      <div ref={containerRef} style={{ flex: 1, background: '#1e1e1e', padding: 4, overflow: 'hidden' }} />
    </div>
  )
}
