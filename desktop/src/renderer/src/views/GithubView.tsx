import React, { useState, useEffect } from 'react'
import { Github, RefreshCw, Loader2, ExternalLink, CheckCircle, XCircle, Copy, FolderGit2, GitPullRequest, AlertCircle } from 'lucide-react'

type Tab = 'auth' | 'repos' | 'issues'
type CliResult = { stdout: string; stderr: string; code: number }

export default function GithubView() {
  const [tab, setTab] = useState<Tab>('auth')
  const [status, setStatus] = useState<CliResult | null>(null)
  const [statusLoading, setStatusLoading] = useState(false)
  const [repos, setRepos] = useState<any[] | null>(null)
  const [reposLoading, setReposLoading] = useState(false)
  const [issues, setIssues] = useState<any[] | null>(null)
  const [issuesLoading, setIssuesLoading] = useState(false)
  const [copied, setCopied] = useState<string | null>(null)

  const loadStatus = async () => {
    setStatusLoading(true)
    try {
      const r = await window.sky.github.cli(['auth', 'status'])
      setStatus(r)
    } finally { setStatusLoading(false) }
  }
  const loadRepos = async () => {
    setReposLoading(true)
    try {
      const r = await window.sky.github.cli(['repo', 'list', '--limit', '50', '--json', 'name,url,description,isPrivate,updatedAt'])
      if (r.code === 0) setRepos(JSON.parse(r.stdout))
      else setRepos([])
    } catch { setRepos([]) }
    finally { setReposLoading(false) }
  }
  const loadIssues = async () => {
    setIssuesLoading(true)
    try {
      const r = await window.sky.github.cli(['issue', 'list', '--limit', '30', '--json', 'number,title,url,state,author,updatedAt,labels'])
      if (r.code === 0) setIssues(JSON.parse(r.stdout))
      else {
        const r2 = await window.sky.github.cli(['pr', 'list', '--limit', '30', '--json', 'number,title,url,state,author,updatedAt'])
        if (r2.code === 0) setIssues(JSON.parse(r2.stdout))
        else setIssues([])
      }
    } catch { setIssues([]) }
    finally { setIssuesLoading(false) }
  }

  useEffect(() => { loadStatus() }, [])

  const copy = async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(key)
      setTimeout(() => setCopied(null), 1200)
    } catch {}
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        {([
          { id: 'auth', label: 'Auth', icon: Github },
          { id: 'repos', label: 'Repositories', icon: FolderGit2 },
          { id: 'issues', label: 'Issues / PRs', icon: GitPullRequest }
        ] as const).map(({ id, label, icon: Icon }) => (
          <div
            key={id}
            onClick={() => setTab(id)}
            style={{
              padding: '0 18px',
              height: 42,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              cursor: 'pointer',
              fontSize: 12,
              borderBottom: tab === id ? '2px solid var(--accent)' : '2px solid transparent',
              color: tab === id ? 'var(--fg-primary)' : 'var(--fg-secondary)',
              marginTop: tab === id ? 0 : 2
            }}
          >
            <Icon size={15} />
            {label}
          </div>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ padding: 8 }}>
          <button
            onClick={() => {
              if (tab === 'auth') loadStatus()
              if (tab === 'repos') loadRepos()
              if (tab === 'issues') loadIssues()
            }}
            style={{ padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {tab === 'auth' && (
          <div style={{ maxWidth: 720 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <Github size={20} />
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>GitHub CLI Authentication</h3>
            </div>
            {statusLoading && <div style={{ color: 'var(--fg-secondary)' }}><Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite', display: 'inline', marginRight: 8 }} />Checking auth status...</div>}
            {!statusLoading && status && (
              <div style={{
                border: '1px solid var(--border-color)', borderRadius: 4, overflow: 'hidden',
                background: 'var(--bg-sidebar)'
              }}>
                <div style={{
                  padding: '12px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  background: status.code === 0 ? 'rgba(78,201,176,0.12)' : 'rgba(244,135,113,0.12)',
                  borderBottom: '1px solid var(--border-color)'
                }}>
                  {status.code === 0 ? <CheckCircle size={18} color="var(--success)" /> : <XCircle size={18} color="var(--error)" />}
                  <span style={{ fontWeight: 600 }}>{status.code === 0 ? 'Authenticated' : 'Not logged in'}</span>
                </div>
                <div style={{ padding: 16 }}>
                  <div style={{ fontSize: 12, color: 'var(--fg-secondary)', marginBottom: 6 }}>STDOUT</div>
                  <pre style={{
                    background: 'var(--bg-editor)',
                    padding: 12,
                    borderRadius: 4,
                    whiteSpace: 'pre-wrap',
                    fontSize: 12,
                    marginBottom: 12,
                    fontFamily: 'Consolas, Monaco, monospace'
                  }}>{status.stdout || '(empty)'}</pre>
                  {status.stderr && (
                    <>
                      <div style={{ fontSize: 12, color: 'var(--fg-secondary)', marginBottom: 6 }}>STDERR</div>
                      <pre style={{
                        background: 'var(--bg-editor)',
                        padding: 12,
                        borderRadius: 4,
                        whiteSpace: 'pre-wrap',
                        fontSize: 12,
                        color: 'var(--error)',
                        fontFamily: 'Consolas, Monaco, monospace'
                      }}>{status.stderr}</pre>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        {tab === 'repos' && (
          <div style={{ maxWidth: 1000 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <FolderGit2 size={18} color="var(--warning)" />
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>Your Repositories</h3>
              {!reposLoading && !repos && <button onClick={loadRepos} style={{ marginLeft: 'auto' }}>Load</button>}
            </div>
            {reposLoading && <div style={{ color: 'var(--fg-secondary)' }}>Loading repos...</div>}
            {!reposLoading && repos && (
              repos.length === 0 ? (
                <div style={{ color: 'var(--fg-secondary)' }}>No repositories. Run `gh auth login` to authenticate.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {repos.map((r, i) => (
                    <div key={i} style={{
                      padding: 12, border: '1px solid var(--border-color)',
                      borderRadius: 4, background: 'var(--bg-sidebar)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                        <Github size={15} color="var(--fg-secondary)" />
                        <span style={{ fontWeight: 600, fontSize: 13 }}>{r.name}</span>
                        {r.isPrivate && <span style={{ fontSize: 10, padding: '1px 5px', background: 'var(--bg-input)', borderRadius: 3 }}>private</span>}
                        <div style={{ flex: 1 }} />
                        <button
                          onClick={() => copy(r.url, `r-${i}-url`)}
                          title="Copy URL"
                          style={{ padding: '2px 6px', background: 'transparent', border: 'none' }}
                        >
                          {copied === `r-${i}-url` ? <CheckCircle size={14} color="var(--success)" /> : <Copy size={14} />}
                        </button>
                        <button
                          onClick={() => window.sky.app.openExternal(r.url)}
                          title="Open in browser"
                          style={{ padding: '2px 6px', background: 'transparent', border: 'none' }}
                        >
                          <ExternalLink size={14} />
                        </button>
                      </div>
                      {r.description && <div style={{ fontSize: 12, color: 'var(--fg-secondary)', marginBottom: 4 }}>{r.description}</div>}
                      {r.updatedAt && <div style={{ fontSize: 11, color: 'var(--fg-secondary)' }}>Updated {r.updatedAt}</div>}
                    </div>
                  ))}
                </div>
              )
            )}
          </div>
        )}
        {tab === 'issues' && (
          <div style={{ maxWidth: 1000 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <AlertCircle size={18} color="var(--error)" />
              <h3 style={{ fontSize: 15, fontWeight: 600 }}>Issues & Pull Requests</h3>
              {!issuesLoading && !issues && <button onClick={loadIssues} style={{ marginLeft: 'auto' }}>Load</button>}
            </div>
            {issuesLoading && <div style={{ color: 'var(--fg-secondary)' }}>Loading issues...</div>}
            {!issuesLoading && issues && (
              issues.length === 0 ? (
                <div style={{ color: 'var(--fg-secondary)' }}>No issues or PRs found.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {issues.map((x, i) => (
                    <div key={i} style={{
                      padding: 12, border: '1px solid var(--border-color)',
                      borderRadius: 4, background: 'var(--bg-sidebar)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span style={{
                          color: x.state === 'OPEN' || x.state === 'open' ? 'var(--success)' : 'var(--fg-secondary)',
                          fontWeight: 700, fontSize: 13
                        }}>
                          #{x.number}
                        </span>
                        <span style={{ fontWeight: 500, fontSize: 13, flex: 1 }}>{x.title}</span>
                        <span style={{ fontSize: 10, padding: '1px 6px', background: 'var(--bg-input)', borderRadius: 3 }}>
                          {x.state}
                        </span>
                        <button
                          onClick={() => copy(x.url, `i-${i}-url`)}
                          title="Copy URL"
                          style={{ padding: '2px 6px', background: 'transparent', border: 'none' }}
                        >
                          {copied === `i-${i}-url` ? <CheckCircle size={14} color="var(--success)" /> : <Copy size={14} />}
                        </button>
                        <button
                          onClick={() => window.sky.app.openExternal(x.url)}
                          title="Open"
                          style={{ padding: '2px 6px', background: 'transparent', border: 'none' }}
                        >
                          <ExternalLink size={14} />
                        </button>
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--fg-secondary)', marginTop: 4 }}>
                        by @{x.author?.login || 'unknown'} · {x.updatedAt}
                        {x.labels && x.labels.length > 0 && (
                          <span style={{ marginLeft: 10 }}>
                            {x.labels.map((l: any, j: number) => (
                              <span key={j} style={{
                                display: 'inline-block', padding: '1px 5px', marginRight: 4,
                                borderRadius: 8, fontSize: 10,
                                background: l.color ? `#${l.color}33` : 'var(--bg-input)',
                                color: l.color ? `#${l.color}` : 'var(--fg-primary)',
                                border: l.color ? `1px solid #${l.color}55` : '1px solid var(--border-color)'
                              }}>{l.name}</span>
                            ))}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )
            )}
          </div>
        )}
      </div>
    </div>
  )
}
