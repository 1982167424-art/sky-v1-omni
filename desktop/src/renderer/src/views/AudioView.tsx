import React, { useState } from 'react'
import { Volume2, Mic, Loader2, Download, Play, Pause, CheckCircle, ExternalLink, Server } from 'lucide-react'

type Mode = 'tts' | 'asr'

export default function AudioView() {
  const [mode, setMode] = useState<Mode>('tts')

  // TTS state
  const [ttsText, setTtsText] = useState('')
  const [voice, setVoice] = useState('female')
  const [ttsFormat, setTtsFormat] = useState('mp3')
  const [ttsSpeed, setTtsSpeed] = useState(1.0)
  const [ttsLoading, setTtsLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioDuration, setAudioDuration] = useState(0)
  const [ttsError, setTtsError] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const audioRef = React.useRef<HTMLAudioElement>(null)

  // ASR state
  const [asrUrl, setAsrUrl] = useState('')
  const [asrLanguage, setAsrLanguage] = useState('zh')
  const [asrLoading, setAsrLoading] = useState(false)
  const [asrText, setAsrText] = useState('')
  const [asrSegments, setAsrSegments] = useState<any[]>([])
  const [asrError, setAsrError] = useState<string | null>(null)

  const runTTS = async () => {
    if (!ttsText.trim() || ttsLoading) return
    setTtsLoading(true)
    setTtsError(null)
    setAudioUrl(null)
    try {
      const res = await window.sky.api.post('/v1/audio/speech', {
        input: ttsText.trim(),
        voice,
        response_format: ttsFormat,
        speed: ttsSpeed
      })
      if (res.ok && res.data?.url) {
        setAudioUrl(res.data.url)
        setAudioDuration(res.data.duration_ms || 0)
      } else {
        throw new Error(res.error || `HTTP ${res.status}`)
      }
    } catch (e: any) {
      setTtsError(e.message || String(e))
    } finally {
      setTtsLoading(false)
    }
  }

  const runASR = async () => {
    if (!asrUrl.trim() || asrLoading) return
    setAsrLoading(true)
    setAsrError(null)
    setAsrText('')
    setAsrSegments([])
    try {
      const res = await window.sky.api.post('/v1/audio/transcriptions', {
        file_url: asrUrl.trim(),
        language: asrLanguage
      })
      if (res.ok && res.data) {
        setAsrText(res.data.text || '')
        setAsrSegments(res.data.segments || [])
      } else {
        throw new Error(res.error || `HTTP ${res.status}`)
      }
    } catch (e: any) {
      setAsrError(e.message || String(e))
    } finally {
      setAsrLoading(false)
    }
  }

  const togglePlay = () => {
    const a = audioRef.current
    if (!a) return
    if (isPlaying) {
      a.pause()
    } else {
      a.play()
    }
    setIsPlaying(!isPlaying)
  }

  const voices = [
    { id: 'female', label: 'VV（活泼女声）' },
    { id: 'male', label: '云舟（沉稳男声）' },
    { id: 'xiaohe', label: '小荷（甜美女声）' },
    { id: 'xiaotian', label: '小天（磁性男声）' }
  ]

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: 16, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <Volume2 size={20} color="var(--fg-accent)" />
          <span style={{ fontWeight: 600 }}>Audio Studio</span>
          <span style={{ fontSize: 11, color: 'var(--fg-secondary)', marginLeft: 8 }}>
            字节火山豆包 TTS 2.0 (seed-tts-2.0) + ASR (volc.bigasr.auc_turbo)
          </span>
          <div style={{ flex: 1 }} />
          <div style={{ display: 'flex', gap: 4, padding: 2, background: 'var(--bg-editor)', borderRadius: 4, border: '1px solid var(--border-color)' }}>
            <button
              onClick={() => setMode('tts')}
              style={{
                padding: '4px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4,
                background: mode === 'tts' ? 'var(--accent)' : 'transparent',
                color: mode === 'tts' ? '#fff' : 'var(--fg-primary)', border: 'none', borderRadius: 3
              }}
            >
              <Volume2 size={13} /> TTS 语音合成
            </button>
            <button
              onClick={() => setMode('asr')}
              style={{
                padding: '4px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4,
                background: mode === 'asr' ? 'var(--accent)' : 'transparent',
                color: mode === 'asr' ? '#fff' : 'var(--fg-primary)', border: 'none', borderRadius: 3
              }}
            >
              <Mic size={13} /> ASR 语音识别
            </button>
          </div>
        </div>

        {mode === 'tts' ? (
          <>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
                <Server size={13} />
                <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>音色：</span>
                <select
                  value={voice}
                  onChange={(e) => setVoice(e.target.value)}
                  style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
                >
                  {voices.map((v) => <option key={v.id} value={v.id}>{v.label}</option>)}
                </select>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
                <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>格式：</span>
                <select
                  value={ttsFormat}
                  onChange={(e) => setTtsFormat(e.target.value)}
                  style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
                >
                  {['mp3', 'wav', 'pcm', 'ogg_opus'].map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
                <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>语速：</span>
                <input
                  type="range"
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  value={ttsSpeed}
                  onChange={(e) => setTtsSpeed(Number(e.target.value))}
                  style={{ width: 100 }}
                />
                <span style={{ fontSize: 11, color: 'var(--fg-primary)', minWidth: 30 }}>{ttsSpeed.toFixed(1)}x</span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
              <textarea
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runTTS() } }}
                placeholder="输入要合成的文字... (Ctrl+Enter 合成)"
                rows={3}
                style={{ flex: 1, minHeight: 70, padding: '10px 12px' }}
              />
              <button
                onClick={runTTS}
                disabled={ttsLoading || !ttsText.trim()}
                style={{ height: 70, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}
              >
                {ttsLoading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Volume2 size={16} />}
                {ttsLoading ? '合成中…' : '合成语音'}
              </button>
            </div>
          </>
        ) : (
          <>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '3px 10px', border: '1px solid var(--border-color)', borderRadius: 4, background: 'var(--bg-editor)' }}>
                <Server size={13} />
                <span style={{ color: 'var(--fg-secondary)', fontSize: 12 }}>语种：</span>
                <select
                  value={asrLanguage}
                  onChange={(e) => setAsrLanguage(e.target.value)}
                  style={{ background: 'transparent', color: 'var(--fg-primary)', border: 'none', outline: 'none', fontSize: 12 }}
                >
                  <option value="zh">中文</option>
                  <option value="en">English</option>
                  <option value="auto">自动检测</option>
                </select>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
              <input
                value={asrUrl}
                onChange={(e) => setAsrUrl(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runASR() } }}
                placeholder="输入音频文件 URL (wav/mp3)... (Enter 识别)"
                style={{ flex: 1, minHeight: 38, padding: '10px 12px' }}
              />
              <button
                onClick={runASR}
                disabled={asrLoading || !asrUrl.trim()}
                style={{ height: 38, padding: '0 18px', background: 'var(--accent)', border: 'none', color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}
              >
                {asrLoading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Mic size={16} />}
                {asrLoading ? '识别中…' : '开始识别'}
              </button>
            </div>
          </>
        )}
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {(ttsError || asrError) && (
          <div style={{ padding: 12, borderRadius: 4, background: 'rgba(244,135,113,0.1)', border: '1px solid var(--error)', color: 'var(--error)', marginBottom: 16 }}>
            {ttsError || asrError}
          </div>
        )}

        {mode === 'tts' && (
          <>
            {ttsLoading && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--fg-secondary)', marginBottom: 16 }}>
                <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} />
                调用豆包 TTS 2.0 合成中…
              </div>
            )}
            {audioUrl && (
              <div style={{ maxWidth: 700 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <CheckCircle size={16} color="var(--success)" /> 语音合成成功
                  {audioDuration > 0 && <span style={{ color: 'var(--fg-secondary)', fontSize: 11, fontWeight: 400 }}>· 时长 {(audioDuration / 1000).toFixed(1)}s</span>}
                </div>
                <div style={{ background: 'var(--bg-sidebar)', border: '1px solid var(--border-color)', borderRadius: 6, padding: 16 }}>
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onEnded={() => setIsPlaying(false)}
                    controls
                    style={{ width: '100%' }}
                  />
                  <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                    <button onClick={togglePlay} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                      {isPlaying ? '暂停' : '播放'}
                    </button>
                    {audioUrl.startsWith('data:') ? (
                      <a
                        href={audioUrl}
                        download={`tts-${Date.now()}.${ttsFormat}`}
                        style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', background: 'var(--bg-input)', color: 'var(--fg-primary)', border: '1px solid var(--border-color)', borderRadius: 4, textDecoration: 'none', fontSize: 13 }}
                      >
                        <Download size={14} /> 下载音频
                      </a>
                    ) : (
                      <button onClick={() => window.sky.app.openExternal(audioUrl)} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <ExternalLink size={14} /> 新窗口打开
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
            {!audioUrl && !ttsLoading && !ttsError && (
              <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60, lineHeight: 1.7 }}>
                输入文字并选择音色，点击「合成语音」生成音频。<br />
                接入 <strong style={{ color: 'var(--fg-primary)' }}>字节火山豆包语音合成大模型 2.0 (seed-tts-2.0)</strong>。<br />
                <span style={{ fontSize: 11 }}>未配置 API Key 时返回模拟音频，便于联调。</span>
              </div>
            )}
          </>
        )}

        {mode === 'asr' && (
          <>
            {asrLoading && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--fg-secondary)', marginBottom: 16 }}>
                <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} />
                调用豆包录音文件识别极速版 (volc.bigasr.auc_turbo)，通常 10s 内出结果…
              </div>
            )}
            {asrText && (
              <div style={{ maxWidth: 800 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <CheckCircle size={16} color="var(--success)" /> 识别结果
                </div>
                <div style={{ background: 'var(--bg-sidebar)', border: '1px solid var(--border-color)', borderRadius: 6, padding: 16, marginBottom: 16 }}>
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{asrText}</div>
                </div>
                {asrSegments.length > 0 && (
                  <>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg-secondary)', marginBottom: 8 }}>分段详情</div>
                    <div style={{ background: 'var(--bg-sidebar)', border: '1px solid var(--border-color)', borderRadius: 6, overflow: 'hidden' }}>
                      {asrSegments.map((s, i) => (
                        <div key={i} style={{ display: 'flex', gap: 12, padding: '8px 12px', borderBottom: i < asrSegments.length - 1 ? '1px solid var(--border-color)' : 'none', fontSize: 12 }}>
                          <span style={{ color: 'var(--fg-secondary)', minWidth: 80 }}>
                            {s.start != null ? `${(s.start / 1000).toFixed(1)}s` : `#${i + 1}`}
                            {s.end != null ? ` - ${(s.end / 1000).toFixed(1)}s` : ''}
                          </span>
                          <span style={{ flex: 1 }}>{s.text}</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
            {!asrText && !asrLoading && !asrError && (
              <div style={{ color: 'var(--fg-secondary)', textAlign: 'center', marginTop: 60, lineHeight: 1.7 }}>
                输入音频文件 URL，点击「开始识别」转录文字。<br />
                接入 <strong style={{ color: 'var(--fg-primary)' }}>字节火山豆包录音文件识别极速版 (volc.bigasr.auc_turbo)</strong>。<br />
                <span style={{ fontSize: 11 }}>支持中英文，单文件 ≤ 2 小时，30 分钟音频约 10s 返回。</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
