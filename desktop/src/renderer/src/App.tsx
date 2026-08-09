import React from 'react'
import {
  MessageSquare,
  Search,
  BrainCircuit,
  BookOpen,
  Github,
  Terminal as TerminalIcon,
  Settings,
  X,
  ChevronDown,
  Power,
  RefreshCw,
  Circle
} from 'lucide-react'
import { useApp, ActivityId } from './store'
import clsx from 'clsx'
import ChatView from './views/ChatView'
import SearchView from './views/SearchView'
import ReasoningView from './views/ReasoningView'
import RagView from './views/RagView'
import GithubView from './views/GithubView'
import TerminalView from './views/TerminalView'
import SettingsView from './views/SettingsView'

const ACTIVITY_ITEMS: { id: ActivityId; icon: React.ComponentType<any>; label: string }[] = [
  { id: 'chat', icon: MessageSquare, label: 'Chat' },
  { id: 'search', icon: Search, label: 'Web Search' },
  { id: 'reasoning', icon: BrainCircuit, label: 'Deep Thinking' },
  { id: 'rag', icon: BookOpen, label: 'RAG / Docs' },
  { id: 'github', icon: Github, label: 'GitHub' },
  { id: 'terminal', icon: TerminalIcon, label: 'Terminal' },
  { id: 'settings', icon: Settings, label: 'Settings' }
]

function ActivityBar() {
  const { state, openActivity } = useApp()
  return (
    <div style={{
      width: 48,
      background: 'var(--bg-activity)',
      display: 'flex',
      flexDirection: 'column',
      borderRight: '1px solid var(--border-color)'
    }}>
      {ACTIVITY_ITEMS.map(({ id, icon: Icon, label }) => {
        const active = state.activeActivity === id
        return (
          <div
            key={id}
            title={label}
            onClick={() => openActivity(id)}
            style={{
              height: 48,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              position: 'relative',
              color: active ? 'var(--fg-primary)' : 'var(--fg-secondary)',
              background: active ? 'rgba(255,255,255,0.05)' : 'transparent'
            }}
          >
            {active && <div style={{
              position: 'absolute',
              left: 0, top: 0, bottom: 0,
              width: 2,
              background: 'var(--accent)'
            }} />}
            <Icon size={22} strokeWidth={active ? 2 : 1.5} />
          </div>
        )
      })}
      <div style={{ flex: 1 }} />
      <div style={{ height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Circle size={14} fill={state.backendStatus.running ? 'var(--success)' : 'var(--error)'} stroke="none" />
      </div>
    </div>
  )
}

function TabBar() {
  const { state, selectTab, closeTab } = useApp()
  return (
    <div style={{
      height: 35,
      background: 'var(--bg-tab)',
      display: 'flex',
      alignItems: 'center',
      borderBottom: '1px solid var(--border-color)',
      overflowX: 'auto'
    }}>
      {state.tabs.map((tab) => {
        const active = tab.id === state.activeTabId
        return (
          <div
            key={tab.id}
            onClick={() => selectTab(tab.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '0 12px',
              height: '100%',
              minWidth: 120,
              cursor: 'pointer',
              background: active ? 'var(--bg-tab-active)' : 'transparent',
              color: active ? 'var(--fg-primary)' : 'var(--fg-secondary)',
              borderRight: '1px solid var(--border-color)',
              borderTop: active ? '1px solid var(--accent)' : '1px solid transparent',
              marginTop: active ? 0 : 1,
              fontSize: 12,
              position: 'relative'
            }}
          >
            <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {tab.title}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); closeTab(tab.id) }}
              style={{
                background: 'transparent',
                border: 'none',
                padding: 2,
                color: 'inherit',
                borderRadius: 3,
                display: 'flex',
                opacity: 0.6
              }}
              onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.6')}
            >
              <X size={14} />
            </button>
          </div>
        )
      })}
    </div>
  )
}

function SidePanel() {
  const { state } = useApp()
  const titles: Record<ActivityId, string> = {
    chat: 'Chat Sessions',
    search: 'Search',
    reasoning: 'Reasoning',
    rag: 'Documents',
    github: 'GitHub',
    terminal: 'Terminal',
    settings: 'Settings'
  }
  return (
    <div style={{
      width: 260,
      background: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-color)',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <div style={{
        height: 35,
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        borderBottom: '1px solid var(--border-color)',
        fontSize: 11,
        textTransform: 'uppercase',
        color: 'var(--fg-secondary)',
        fontWeight: 600,
        letterSpacing: 0.5
      }}>
        {titles[state.activeActivity]}
        <ChevronDown size={14} style={{ marginLeft: 'auto' }} />
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
        {state.activeActivity === 'chat' && (
          <div style={{ color: 'var(--fg-secondary)', fontSize: 12, padding: 8 }}>
            No saved sessions yet. Start a new chat from the main panel.
          </div>
        )}
        {state.activeActivity === 'search' && (
          <div style={{ color: 'var(--fg-secondary)', fontSize: 12, padding: 8 }}>
            Configure search providers and filters.
          </div>
        )}
        {state.activeActivity === 'rag' && (
          <div style={{ color: 'var(--fg-secondary)', fontSize: 12, padding: 8 }}>
            Manage documents and collections.
          </div>
        )}
        {!['chat', 'search', 'rag'].includes(state.activeActivity) && (
          <div style={{ color: 'var(--fg-secondary)', fontSize: 12, padding: 8 }}>
            No explorer content for this view.
          </div>
        )}
      </div>
    </div>
  )
}

function ActivityRenderer() {
  const { state } = useApp()
  const tab = state.tabs.find((t) => t.id === state.activeTabId)
  const activity = tab?.activity || state.activeActivity
  switch (activity) {
    case 'chat': return <ChatView />
    case 'search': return <SearchView />
    case 'reasoning': return <ReasoningView />
    case 'rag': return <RagView />
    case 'github': return <GithubView />
    case 'terminal': return <TerminalView />
    case 'settings': return <SettingsView />
    default: return <div style={{ padding: 20, color: 'var(--fg-secondary)' }}>Select a view</div>
  }
}

function PanelArea() {
  const { state, toggleTerminal } = useApp()
  if (!state.terminalOpen) return null
  return (
    <div style={{
      height: 260,
      display: 'flex',
      flexDirection: 'column',
      borderTop: '1px solid var(--border-color)',
      background: 'var(--bg-panel)'
    }}>
      <div style={{
        height: 32,
        display: 'flex',
        alignItems: 'center',
        borderBottom: '1px solid var(--border-color)',
        background: 'var(--bg-sidebar)',
        padding: '0 8px',
        gap: 4
      }}>
        <div style={{
          padding: '0 12px',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          fontSize: 11,
          textTransform: 'uppercase',
          color: 'var(--fg-primary)',
          borderTop: '1px solid var(--accent)',
          background: 'var(--bg-panel)'
        }}>
          <TerminalIcon size={13} style={{ marginRight: 6 }} />
          Terminal
        </div>
        <div style={{ flex: 1 }} />
        <button
          onClick={toggleTerminal}
          title="Close Panel"
          style={{
            background: 'transparent',
            border: 'none',
            padding: 4,
            color: 'var(--fg-secondary)',
            borderRadius: 3
          }}
        >
          <X size={14} />
        </button>
      </div>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <TerminalView embedded />
      </div>
    </div>
  )
}

function StatusBar() {
  const { state, dispatch, openActivity } = useApp()
  return (
    <div style={{
      height: 22,
      background: 'var(--accent)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 10px',
      fontSize: 11,
      color: '#fff',
      gap: 16
    }}>
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}
        onClick={async () => {
          await window.sky.backend.restart()
        }}
        title="Click to restart backend"
      >
        <Circle size={10} fill={state.backendStatus.running ? '#89e0a8' : '#ff8c8c'} stroke="none" />
        Backend {state.backendStatus.running ? 'Running' : 'Stopped'} :{state.backendStatus.port}
      </div>
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
        onClick={() => dispatch({ type: 'TOGGLE_TERMINAL' })}
      >
        <TerminalIcon size={12} />
        Terminal
      </div>
      <div style={{ flex: 1 }} />
      <div
        onClick={() => openActivity('github')}
        style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
      >
        <Github size={12} />
        GitHub
      </div>
      <div>v{state.version}</div>
    </div>
  )
}

function MainToolbar() {
  const { toggleTerminal } = useApp()
  return (
    <div style={{
      height: 38,
      background: 'var(--bg-editor)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 12px',
      gap: 8,
      borderBottom: '1px solid var(--border-color)'
    }}>
      <button onClick={() => window.sky.backend.restart()} title="Restart Backend" style={{ padding: '4px 8px', display: 'flex', gap: 6, alignItems: 'center' }}>
        <RefreshCw size={14} /> Restart
      </button>
      <button onClick={() => window.sky.backend.stop()} title="Stop Backend" style={{ padding: '4px 8px', display: 'flex', gap: 6, alignItems: 'center' }}>
        <Power size={14} /> Stop
      </button>
      <div style={{ flex: 1 }} />
      <button onClick={toggleTerminal} title="Toggle Terminal" style={{ padding: '4px 8px', display: 'flex', gap: 6, alignItems: 'center' }}>
        <TerminalIcon size={14} /> Toggle Terminal
      </button>
    </div>
  )
}

export default function App() {
  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'row' }}>
      <ActivityBar />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <TabBar />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'row', overflow: 'hidden' }}>
          <SidePanel />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <MainToolbar />
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <ActivityRenderer />
            </div>
            <PanelArea />
          </div>
        </div>
        <StatusBar />
      </div>
    </div>
  )
}
