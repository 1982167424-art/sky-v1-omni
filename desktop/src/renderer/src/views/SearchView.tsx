import React, { useState } from 'react'
import { Search, ExternalLink, Loader2 } from 'lucide-react'

interface SearchResult {
  title: string
  snippet: string
  url: string
  provider?: string
}

export default function SearchView() {
  const [query, setQuery] = useState('')
  const [providers, setProviders] = useState<string[]>(['google', 'bing'])
  const [num, setNum] = useState(10)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SearchResult[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const providerOptions = ['google', 'baidu', 'bing', 'toutiao']

  const toggleProvider = (p: string) => {
    setProviders((prev) =>
      prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]
    )
  }

  const doSearch = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await window.sky.api.post('/search/web', {
        query: query.trim(),
        providers,
        num
      })
      if (res.ok && res.data) {
        const r = res.data.results || res.data.data || (Array.isArray(res.data) ? res.data : [])
        setResults(r)
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
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14 }}>
          <div style={{ flex: 1, position: 'relative' }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && doSearch()}
              placeholder="Search the web..."
              style={{ width: '100%', height: 36, paddingLeft: 36 }}
            />
            <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--fg-secondary)' }} />
          </div>
          <button
            onClick={doSearch}
            disabled={loading || !query.trim()}
            style={{ height: 36, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff' }}
          >
            {loading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : 'Search'}
          </button>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>Providers:</span>
            {providerOptions.map((p) => (
              <label
                key={p}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '3px 8px',
                  background: providers.includes(p) ? 'var(--accent)' : 'var(--bg-input)',
                  color: providers.includes(p) ? '#fff' : 'var(--fg-primary)',
                  borderRadius: 4,
                  fontSize: 12,
                  cursor: 'pointer',
                  userSelect: 'none'
                }}
              >
                <input
                  type="checkbox"
                  checked={providers.includes(p)}
                  onChange={() => toggleProvider(p)}
                  style={{ display: 'none' }}
                />
                {p}
              </label>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>Results: {num}</span>
            <input
              type="range"
              min={5}
              max={50}
              step={5}
              value={num}
              onChange={(e) => setNum(Number(e.target.value))}
              style={{ width: 120 }}
            />
          </div>
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
        {loading && <div style={{ color: 'var(--fg-secondary)' }}>Searching...</div>}
        {!loading && results && (
          results.length === 0 ? (
            <div style={{ color: 'var(--fg-secondary)' }}>No results found.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 860 }}>
              {results.map((r, i) => (
                <div key={i} style={{
                  padding: 14, borderRadius: 4,
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 4 }}>
                    <a
                      onClick={() => window.sky.app.openExternal(r.url)}
                      style={{ color: 'var(--fg-accent)', textDecoration: 'none', fontWeight: 500, fontSize: 15, cursor: 'pointer' }}
                    >
                      {r.title || '(no title)'}
                    </a>
                    <ExternalLink size={13} style={{ color: 'var(--fg-secondary)', marginTop: 4, flexShrink: 0 }} />
                  </div>
                  <div style={{ color: 'var(--success)', fontSize: 12, marginBottom: 6 }}>
                    {r.url}
                  </div>
                  <div style={{ color: 'var(--fg-primary)', fontSize: 13, lineHeight: 1.5 }}>
                    {r.snippet}
                  </div>
                  {r.provider && (
                    <div style={{ marginTop: 6, display: 'inline-block', padding: '1px 6px', background: 'var(--bg-input)', borderRadius: 3, fontSize: 11, color: 'var(--fg-secondary)' }}>
                      {r.provider}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )
        )}
        {!loading && !results && !error && (
          <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60 }}>
            Enter a query and click Search to begin.
          </div>
        )}
      </div>
    </div>
  )
}
