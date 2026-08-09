import React, { createContext, useContext, useReducer, useEffect } from 'react'

export type ActivityId = 'chat' | 'search' | 'reasoning' | 'rag' | 'github' | 'terminal' | 'image' | 'video' | '3d' | 'audio' | 'settings'

export interface Tab {
  id: string
  title: string
  activity: ActivityId
}

export interface AppState {
  tabs: Tab[]
  activeTabId: string | null
  activeActivity: ActivityId
  backendStatus: { running: boolean; port: number }
  terminalOpen: boolean
  version: string
  logs: string[]
}

type Action =
  | { type: 'SET_ACTIVITY'; payload: ActivityId }
  | { type: 'OPEN_TAB'; payload: Tab }
  | { type: 'CLOSE_TAB'; payload: string }
  | { type: 'SET_ACTIVE_TAB'; payload: string }
  | { type: 'SET_BACKEND_STATUS'; payload: { running: boolean; port: number } }
  | { type: 'TOGGLE_TERMINAL' }
  | { type: 'SET_TERMINAL_OPEN'; payload: boolean }
  | { type: 'SET_VERSION'; payload: string }
  | { type: 'APPEND_LOG'; payload: string }
  | { type: 'SET_LOGS'; payload: string[] }

const initialState: AppState = {
  tabs: [{ id: 'tab-chat', title: 'Chat', activity: 'chat' }],
  activeTabId: 'tab-chat',
  activeActivity: 'chat',
  backendStatus: { running: false, port: 8765 },
  terminalOpen: false,
  version: '0.1.0',
  logs: []
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_ACTIVITY': {
      const activity = action.payload
      const existingTab = state.tabs.find((t) => t.activity === activity)
      let tabs = state.tabs
      let activeTabId = state.activeTabId
      if (existingTab) {
        activeTabId = existingTab.id
      } else {
        const titles: Record<ActivityId, string> = {
          chat: 'Chat',
          search: 'Search',
          reasoning: 'Deep Thinking',
          rag: 'RAG / Docs',
          github: 'GitHub',
          terminal: 'Terminal',
          image: 'Image Gen',
          video: 'Video Gen',
          '3d': '3D Viewer',
          audio: 'Audio',
          settings: 'Settings'
        }
        const newTab: Tab = {
          id: `tab-${activity}-${Date.now()}`,
          title: titles[activity],
          activity
        }
        tabs = [...tabs, newTab]
        activeTabId = newTab.id
      }
      return { ...state, activeActivity: activity, tabs, activeTabId }
    }
    case 'OPEN_TAB': {
      const exists = state.tabs.some((t) => t.id === action.payload.id)
      return {
        ...state,
        tabs: exists ? state.tabs : [...state.tabs, action.payload],
        activeTabId: action.payload.id,
        activeActivity: action.payload.activity
      }
    }
    case 'CLOSE_TAB': {
      const remaining = state.tabs.filter((t) => t.id !== action.payload)
      let activeTabId = state.activeTabId
      if (activeTabId === action.payload) {
        activeTabId = remaining.length > 0 ? remaining[remaining.length - 1].id : null
      }
      const activeActivity = activeTabId
        ? remaining.find((t) => t.id === activeTabId)?.activity || state.activeActivity
        : state.activeActivity
      return { ...state, tabs: remaining, activeTabId, activeActivity }
    }
    case 'SET_ACTIVE_TAB': {
      const tab = state.tabs.find((t) => t.id === action.payload)
      return {
        ...state,
        activeTabId: action.payload,
        activeActivity: tab?.activity || state.activeActivity
      }
    }
    case 'SET_BACKEND_STATUS':
      return { ...state, backendStatus: action.payload }
    case 'TOGGLE_TERMINAL':
      return { ...state, terminalOpen: !state.terminalOpen }
    case 'SET_TERMINAL_OPEN':
      return { ...state, terminalOpen: action.payload }
    case 'SET_VERSION':
      return { ...state, version: action.payload }
    case 'APPEND_LOG': {
      const next = [...state.logs, action.payload]
      return { ...state, logs: next.slice(-200) }
    }
    case 'SET_LOGS':
      return { ...state, logs: action.payload.slice(-200) }
    default:
      return state
  }
}

interface AppContextValue {
  state: AppState
  dispatch: React.Dispatch<Action>
  openActivity: (a: ActivityId) => void
  closeTab: (id: string) => void
  selectTab: (id: string) => void
  toggleTerminal: () => void
}

const AppContext = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    window.sky?.app?.getVersion().then((v) => dispatch({ type: 'SET_VERSION', payload: v })).catch(() => {})
    window.sky?.backend?.status().then((s) => {
      dispatch({ type: 'SET_BACKEND_STATUS', payload: { running: s.running, port: s.port } })
      dispatch({ type: 'SET_LOGS', payload: s.logs })
    }).catch(() => {})
    const offLog = window.sky?.backend?.onLog?.((line) => dispatch({ type: 'APPEND_LOG', payload: line }))
    const offStatus = window.sky?.backend?.onStatusChange?.((s) =>
      dispatch({ type: 'SET_BACKEND_STATUS', payload: s })
    )
    return () => {
      offLog?.()
      offStatus?.()
    }
  }, [])

  const value: AppContextValue = {
    state,
    dispatch,
    openActivity: (a) => dispatch({ type: 'SET_ACTIVITY', payload: a }),
    closeTab: (id) => dispatch({ type: 'CLOSE_TAB', payload: id }),
    selectTab: (id) => dispatch({ type: 'SET_ACTIVE_TAB', payload: id }),
    toggleTerminal: () => dispatch({ type: 'TOGGLE_TERMINAL' })
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
