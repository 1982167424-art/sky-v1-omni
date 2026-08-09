import React, { useEffect, useState } from 'react'
import { BookOpen, Upload, Search, Loader2, FileText, CheckCircle } from 'lucide-react'

interface RagResult {
  id?: string
  score?: number
  content: string
  source?: string
}

export default function RagView() {
  const [files, setFiles] = useState<string[]>([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [ingesting, setIngesting] = useState<string | null>(null)
  const [ingestMsg, setIngestMsg] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [searching, setSearching] = useState(false)
  const [results, setResults] = useState<RagResult[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadFiles = async () => {
    setLoadingFiles(true)
    try {
      const r = await window.sky.rag.listFiles()
      if (r.ok) setFiles(r.files)
      else throw new Error(r.error || 'Failed to list')
    } catch (e: any) { setError(e.message) }
    finally { setLoadingFiles(false) }
  }

  useEffect(() => {
    loadFiles()
  }, [])

  const ingestFile = async (f: string) => {
    setIngesting(f)
    setIngestMsg(null)
    try {
      const r = await window.sky.rag.ingest(f)
      if (r.ok) setIngestMsg(`Ingested ${f} successfully.`)
      else throw new Error(r.error || 'Failed')
    } catch (e: any) {
      setIngestMsg(`Error: ${e.message}`)
    } finally { setIngesting(null) }
  }

  const doQuery = async () => {
    if (!query.trim() || searching) return
    setSearching(true)
    setError(null)
    setResults(null)
    try {
      const r = await window.sky.api.post('/rag/query', { query: query.trim(), top_k: topK })
      if (r.ok && r.data) {
        const raw = r.data.results || r.data.documents || r.data.data || (Array.isArray(r.data) ? r.data : [])
        setResults(raw)
      } else throw new Error(r.error || `HTTP ${r.status}`)
    } catch (e: any) { setError(e.message) }
    finally { setSearching(false) }
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: 16, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <BookOpen size={20} color="var(--warning)" />
          <span style={{ fontWeight: 600 }}>RAG / Documents</span>
          <div style={{ flex: 1 }} />
          <button onClick={loadFiles} disabled={loadingFiles} style={{ padding: '4px 10px' }}>
            {loadingFiles ? <Loader2 size={14} style={{ animation: 'spin 0.8s linear infinite' }} /> : 'Refresh'}
          </button>
        </div>
        <div style={{ fontSize: 12, color: 'var(--fg-secondary)', marginBottom: 8 }}>Preset files (sky_v1/rag/presets)</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
          {loadingFiles && <span style={{ color: 'var(--fg-secondary)' }}>Loading...</span>}
          {!loadingFiles && files.length === 0 && <span style={{ color: 'var(--fg-secondary)' }}>No preset files found.</span>}
          {files.map((f) => (
            <div key={f} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '6px 10px', background: 'var(--bg-editor)',
              border: '1px solid var(--border-color)', borderRadius: 4
            }}>
              <FileText size={14} color="var(--fg-accent)" />
              <span style={{ fontSize: 12 }}>{f}</span>
              <button
                onClick={() => ingestFile(f)}
                disabled={ingesting === f}
                style={{
                  padding: '2px 6px', fontSize: 11,
                  background: 'var(--accent)', border: 'none', color: '#fff',
                  display: 'flex', alignItems: 'center', gap: 4
                }}
              >
                {ingesting === f ? <Loader2 size={11} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Upload size={11} />}
                Ingest
              </button>
            </div>
          ))}
        </div>
        {ingestMsg && (
          <div style={{ padding: '6px 10px', borderRadius: 4, fontSize: 12, background: ingestMsg.startsWith('Error') ? 'rgba(244,135,113,0.1)' : 'rgba(78,201,176,0.1)', color: ingestMsg.startsWith('Error') ? 'var(--error)' : 'var(--success)', marginBottom: 10 }}>
            {ingestMsg.startsWith('Error') ? ingestMsg : <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><CheckCircle size={14} />{ingestMsg}</span>}
          </div>
        )}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{ flex: 1, position: 'relative' }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && doQuery()}
              placeholder="Query ingested documents..."
              style={{ width: '100%', height: 36, paddingLeft: 36 }}
            />
            <Search size={16} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--fg-secondary)' }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <span style={{ color: 'var(--fg-secondary)' }}>Top K: {topK}</span>
            <input type="range" min={1} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: 100 }} />
          </div>
          <button onClick={doQuery} disabled={searching || !query.trim()} style={{ height: 36, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff' }}>
            {searching ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : 'Query'}
          </button>
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {error && (
          <div style={{ padding: 12, borderRadius: 4, background: 'rgba(244,135,113,0.1)', border: '1px solid var(--error)', color: 'var(--error)', marginBottom: 16 }}>
            Error: {error}
          </div>
        )}
        {searching && <div style={{ color: 'var(--fg-secondary)' }}>Searching knowledge base...</div>}
        {!searching && results && (
          results.length === 0 ? (
            <div style={{ color: 'var(--fg-secondary)' }}>No matching documents found.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 900 }}>
              {results.map((r, i) => (
                <div key={i} style={{
                  padding: 14, borderRadius: 4, background: 'rgba(255,255,255,0.02)',
                  border: '1px solid var(--border-color)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div style={{ fontSize: 11, color: 'var(--fg-secondary)' }}>
                      Result {i + 1}{r.score !== undefined && ` · relevance ${(r.score as number).toFixed(3)}`}
                    </div>
                    {r.source && (
                      <span style={{ padding: '1px 8px', fontSize: 11, background: 'var(--bg-input)', borderRadius: 3, color: 'var(--fg-accent)' }}>
                        {r.source}
                      </span>
                    )}
                  </div>
                  <div style={{ lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{r.content}</div>
                </div>
              ))}
            </div>
          )
        )}
        {!searching && !results && !error && (
          <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60, lineHeight: 1.7 }}>
            Select preset files to ingest, then query the knowledge base.
          </div>
        )}
      </div>
    </div>
  )
}
