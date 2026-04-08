import { useEffect, useMemo, useState } from 'react'

const FEATURE_META = {
  ack_count: { label: 'ACK Packet Count', description: 'Acknowledgement packets seen in this flow window.' },
  avg: { label: 'Average Value', description: 'Average packet statistic for the current flow window.' },
  covariance: { label: 'Covariance', description: 'Joint variation between paired packet statistics.' },
  cwr_flag_number: { label: 'CWR Flag Count', description: 'Congestion Window Reduced TCP flags observed.' },
  drate: { label: 'Destination Packet Rate', description: 'Estimated packets per second received by destination.' },
  ece_flag_number: { label: 'ECE Flag Count', description: 'ECN-Echo TCP flags observed.' },
  fin_count: { label: 'FIN Packet Count', description: 'Connection-closing FIN packets observed.' },
  fin_flag_number: { label: 'FIN Flag Number', description: 'Feature-engineered FIN flag count indicator.' },
  header_length: { label: 'Header Length', description: 'Packet header size information in the flow window.' },
  iat: { label: 'Inter-Arrival Time (IAT)', description: 'Time gap between packets.' },
  magnitue: { label: 'Magnitude', description: 'Composite traffic magnitude feature from the dataset.' },
  max: { label: 'Maximum Value', description: 'Maximum packet statistic inside this flow window.' },
  min: { label: 'Minimum Value', description: 'Minimum packet statistic inside this flow window.' },
  number: { label: 'Packet Count', description: 'Number of packets observed in this flow window.' },
  psh_flag_number: { label: 'PSH Flag Count', description: 'TCP PUSH flags observed.' },
  radius: { label: 'Radius', description: 'Spread feature derived from the traffic distribution.' },
  rate: { label: 'Packet Rate', description: 'Estimated packets per second in the flow window.' },
  rst_count: { label: 'RST Packet Count', description: 'Connection reset packets observed.' },
  rst_flag_number: { label: 'RST Flag Number', description: 'Feature-engineered RST flag count indicator.' },
  srate: { label: 'Source Packet Rate', description: 'Estimated packets per second sent by source.' },
  std: { label: 'Standard Deviation', description: 'Variation level of packet statistics in this flow window.' },
  syn_count: { label: 'SYN Packet Count', description: 'Connection-open SYN packets observed.' },
  syn_flag_number: { label: 'SYN Flag Number', description: 'Feature-engineered SYN flag count indicator.' },
  tot_size: { label: 'Total Bytes', description: 'Total packet bytes transferred in this flow window.' },
  tot_sum: { label: 'Total Sum', description: 'Aggregate flow-level sum feature from the dataset.' },
  variance: { label: 'Variance', description: 'Variance of packet statistics in the flow window.' },
  weight: { label: 'Weight', description: 'Composite weighted traffic feature from the dataset.' },
}

const PROTOCOL_FILTERS = ['all', 'wifi', 'mqtt', 'bluetooth']

function safeNumber(value, fallback = 0) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function formatSimClock(seconds) {
  const total = Math.max(0, Math.floor(seconds))
  const h = String(Math.floor(total / 3600)).padStart(2, '0')
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0')
  const s = String(total % 60).padStart(2, '0')
  return `${h}:${m}:${s}`
}

function formatTime(value) {
  if (!value) return 'n/a'
  try {
    return new Date(value).toLocaleTimeString()
  } catch {
    return 'n/a'
  }
}

function normalizeFeatureKey(feature) {
  return String(feature || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function fallbackFeatureLabel(feature) {
  return String(feature || '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function getFeatureMeta(feature) {
  const key = normalizeFeatureKey(feature)
  const meta = FEATURE_META[key]
  if (meta) return meta
  return {
    label: fallbackFeatureLabel(feature),
    description: 'Flow-level feature used by the CatBoost model.',
  }
}

function directionText(direction) {
  if (direction === 'attack') return 'pushes this decision toward attack'
  if (direction === 'benign') return 'pulls this decision toward benign'
  return 'influences this decision'
}

function normalizeFilenameBit(value) {
  return String(value || 'value').toLowerCase().replace(/[^a-z0-9_-]+/g, '_')
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function buildAlertNarrative(alert) {
  if (!alert) return ''
  const top = (alert.local_explanation || []).slice(0, 3).map((f) => getFeatureMeta(f.feature).label)
  const score = safeNumber(alert.score_attack, 0)
  const threshold = safeNumber(alert.threshold, 0.5)
  const family = String(alert.attack_family || '').trim()

  let text = `Flagged because attack score ${score.toFixed(4)} is above the ${String(alert.protocol || '').toUpperCase()} threshold ${threshold.toFixed(4)}.`
  if (top.length > 0) text += ` Main drivers: ${top.join(', ')}.`
  if (family && family.toLowerCase() !== 'n/a') text += ` Predicted family context: ${family}.`
  return text
}

function buildAlertEvidence(alert, meta, state) {
  if (!alert) return null
  const score = safeNumber(alert.score_attack, 0)
  const threshold = safeNumber(alert.threshold, 0.5)
  return {
    evidence_version: 'thesis_ui_v1',
    exported_at_utc: new Date().toISOString(),
    model_family: String(meta?.model_family || 'CatBoost-E'),
    threshold_policy: String(meta?.threshold_policy || 'n/a'),
    replay_mode: String(meta?.replay_mode || 'simulator_replay'),
    replay_order: String(meta?.replay_order || 'n/a'),
    rows_per_second: safeNumber(state?.rows_per_second, safeNumber(meta?.rows_per_second, 5)),
    alert: {
      ...alert,
      decision_margin: score - threshold,
    },
  }
}

async function apiGet(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

async function apiPost(path, body) {
  const opts = { method: 'POST' }
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' }
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(path, opts)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

function App() {
  const [meta, setMeta] = useState(null)
  const [globalExplanations, setGlobalExplanations] = useState(null)
  const [state, setState] = useState(null)
  const [apiStatus, setApiStatus] = useState('degraded')
  const [apiLastRefreshAt, setApiLastRefreshAt] = useState('')
  const [selectedAlertId, setSelectedAlertId] = useState('')
  const [protocolFilter, setProtocolFilter] = useState('all')
  const [attackFamilyQuery, setAttackFamilyQuery] = useState('')
  const [copyStatus, setCopyStatus] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    async function bootstrap() {
      try {
        const initPayload = await apiGet('/api/init')
        if (!active) return
        const nextMeta = initPayload.meta || {}
        const nextState = initPayload.state || {}
        setMeta(nextMeta)
        setGlobalExplanations(initPayload.global_explanations || {})
        setState(nextState)
        setApiStatus('healthy')
        setApiLastRefreshAt(new Date().toISOString())
        setError('')
      } catch (e) {
        if (!active) return
        setApiStatus('degraded')
        setError(`Failed to initialize realtime API: ${String(e?.message || e)}`)
      }
    }
    bootstrap()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!meta) return undefined
    let active = true

    const pollState = async () => {
      try {
        const payload = await apiGet('/api/state')
        if (!active) return
        setState(payload.state || {})
        setApiStatus('healthy')
        setApiLastRefreshAt(new Date().toISOString())
      } catch (e) {
        if (!active) return
        setApiStatus('degraded')
        setError(`Realtime API connection failed: ${String(e?.message || e)}`)
      }
    }

    const pollHealth = async () => {
      try {
        await apiGet('/api/health')
        if (!active) return
        setApiStatus('healthy')
      } catch {
        if (!active) return
        setApiStatus('degraded')
      }
    }

    pollState()
    pollHealth()

    const stateTimer = setInterval(pollState, 1000)
    const healthTimer = setInterval(pollHealth, 5000)

    return () => {
      active = false
      clearInterval(stateTimer)
      clearInterval(healthTimer)
    }
  }, [meta])

  const recentAlerts = useMemo(() => state?.recent_alerts || [], [state])

  const filteredAlerts = useMemo(() => {
    const q = attackFamilyQuery.trim().toLowerCase()
    return recentAlerts.filter((alert) => {
      const protocolOk = protocolFilter === 'all' || String(alert.protocol || '').toLowerCase() === protocolFilter
      const family = String(alert.attack_family || '').toLowerCase()
      const familyOk = q.length === 0 || family.includes(q)
      return protocolOk && familyOk
    })
  }, [recentAlerts, protocolFilter, attackFamilyQuery])

  useEffect(() => {
    if (filteredAlerts.length === 0) {
      setSelectedAlertId('')
      return
    }
    if (!selectedAlertId || !filteredAlerts.some((a) => a.id === selectedAlertId)) {
      setSelectedAlertId(filteredAlerts[0].id)
    }
  }, [selectedAlertId, filteredAlerts])

  useEffect(() => {
    if (!copyStatus) return undefined
    const timer = setTimeout(() => setCopyStatus(''), 1800)
    return () => clearTimeout(timer)
  }, [copyStatus])

  const selectedAlert = useMemo(() => {
    return filteredAlerts.find((a) => a.id === selectedAlertId) || null
  }, [selectedAlertId, filteredAlerts])

  const selectedMaxAbs = useMemo(() => {
    const vals = (selectedAlert?.local_explanation || []).map((f) => safeNumber(f.abs_contribution, 0))
    return vals.length > 0 ? Math.max(...vals) : 1
  }, [selectedAlert])

  const toggleRunning = async () => {
    try {
      const payload = state?.running ? await apiPost('/api/pause') : await apiPost('/api/start')
      setState(payload.state || {})
      setApiStatus('healthy')
      setApiLastRefreshAt(new Date().toISOString())
      setError('')
    } catch (e) {
      setApiStatus('degraded')
      setError(`Failed to change run state: ${String(e?.message || e)}`)
    }
  }

  const reset = async () => {
    try {
      const payload = await apiPost('/api/reset')
      setState(payload.state || {})
      setSelectedAlertId('')
      setError('')
    } catch (e) {
      setError(`Failed to reset: ${String(e?.message || e)}`)
    }
  }

  const copySelectedAlert = async () => {
    if (!selectedAlert) return
    const payload = buildAlertEvidence(selectedAlert, meta, state)
    if (!payload) return
    const text = JSON.stringify(payload, null, 2)
    try {
      if (!navigator?.clipboard?.writeText) throw new Error('Clipboard API unavailable')
      await navigator.clipboard.writeText(text)
      setCopyStatus('Evidence copied')
      setError('')
    } catch (e) {
      setError(`Failed to copy evidence JSON: ${String(e?.message || e)}`)
    }
  }

  const downloadSelectedAlert = () => {
    if (!selectedAlert) return
    const payload = buildAlertEvidence(selectedAlert, meta, state)
    if (!payload) return
    const filename = `alert_evidence_${normalizeFilenameBit(selectedAlert.protocol)}_${safeNumber(selectedAlert.global_row_index, 0)}.json`
    downloadText(filename, JSON.stringify(payload, null, 2), 'application/json;charset=utf-8')
  }

  if (error && !state) {
    return (
      <div className="app-shell">
        <h1>IDS Operations Console</h1>
        <p className="error">{error}</p>
      </div>
    )
  }

  if (!meta || !state || !globalExplanations) {
    return (
      <div className="app-shell">
        <h1>IDS Operations Console</h1>
        <p>Connecting to CatBoost-E realtime API...</p>
      </div>
    )
  }

  const flowsProcessed = safeNumber(state.flows_processed, 0)
  const alertsSurfaced = safeNumber(state.alerts_surfaced, 0)

  const margin = selectedAlert
    ? safeNumber(selectedAlert.score_attack, 0) - safeNumber(selectedAlert.threshold, 0.5)
    : 0

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>IDS Operations Console</h1>
          <p className="subtitle">
            CatBoost-E multi-protocol intrusion monitoring with live explainability.
          </p>
        </div>
        <div className="controls">
          <button className="primary" onClick={toggleRunning} disabled={Boolean(state.ended)}>
            {state.running ? 'Pause Monitoring' : 'Start Monitoring'}
          </button>
          <button className="ghost" onClick={reset}>Clear Session</button>
        </div>
      </header>

      <section className="card context-strip">
        <div className="context-grid">
          <div><span className="label">Model</span><span className="context-value">{String(meta.model_family || 'CatBoost-E')}</span></div>
          <div><span className="label">Threshold Policy</span><span className="context-value">{String(meta.threshold_policy || 'n/a')}</span></div>
          <div><span className="label">Inference</span><span className="context-value">{String(meta.inference_device_active || 'cpu').toUpperCase()}</span></div>
        </div>
        <div className="api-status-wrap">
          <span className={`status-pill ${apiStatus === 'healthy' ? 'healthy' : 'degraded'}`}>
            API {apiStatus === 'healthy' ? 'healthy' : 'degraded'}
          </span>
          <span className="status-time">Last refresh {formatTime(apiLastRefreshAt)}</span>
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}

      <section className="stats-grid">
        <div className="card stat"><div className="label">Monitoring Uptime</div><div className="value">{formatSimClock(safeNumber(state.sim_seconds, 0))}</div></div>
        <div className="card stat"><div className="label">Events Processed</div><div className="value">{Math.floor(flowsProcessed).toLocaleString()}</div></div>
        <div className="card stat"><div className="label">Alerts Detected</div><div className="value">{Math.floor(alertsSurfaced).toLocaleString()}</div></div>
      </section>

      <section className="main-grid">
        <div className="card panel">
          <h2>Sequential Alerts</h2>
          <div className="triage-controls">
            <div className="chip-row">
              {PROTOCOL_FILTERS.map((proto) => (
                <button
                  key={proto}
                  className={`chip ${protocolFilter === proto ? 'active' : ''}`}
                  onClick={() => setProtocolFilter(proto)}
                >
                  {proto}
                </button>
              ))}
            </div>
            <input
              className="family-search"
              type="text"
              placeholder="Search attack family"
              value={attackFamilyQuery}
              onChange={(e) => setAttackFamilyQuery(e.target.value)}
            />
          </div>
          {filteredAlerts.length === 0 ? (
            <p className="muted">No alerts match the current triage filters.</p>
          ) : (
            <ul className="alert-list">
              {filteredAlerts.map((a) => {
                const itemMargin = safeNumber(a.score_attack, 0) - safeNumber(a.threshold, 0.5)
                return (
                  <li key={a.id}>
                    <button className={`alert-item ${selectedAlertId === a.id ? 'active' : ''}`} onClick={() => setSelectedAlertId(a.id)}>
                      <span className="badge">{a.protocol}</span>
                      <span className="alert-main">score={safeNumber(a.score_attack).toFixed(4)} thr={safeNumber(a.threshold).toFixed(4)} margin={itemMargin.toFixed(4)}</span>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>

        <div className="card panel">
          <h2>Local Explanation</h2>
          {!selectedAlert ? (
            <p className="muted">Select an alert to inspect local feature contributions.</p>
          ) : (
            <>
              <div className="alert-detail-head">
                <span className="badge">{selectedAlert.protocol}</span>
                <span>attack_family: {selectedAlert.attack_family || 'n/a'}</span>
                <span className={`margin-pill ${margin >= 0 ? 'positive' : 'negative'}`}>margin={margin.toFixed(4)}</span>
              </div>
              <div className="explain-note">Interpretation: positive contribution supports attack, negative contribution supports benign.</div>
              <p className="narrative">{buildAlertNarrative(selectedAlert)}</p>
              <div className="evidence-actions">
                <button className="ghost" onClick={copySelectedAlert}>Copy Alert Evidence JSON</button>
                <button className="ghost" onClick={downloadSelectedAlert}>Download Alert JSON</button>
                {copyStatus ? <span className="copy-status">{copyStatus}</span> : null}
              </div>
              <ul className="feature-list">
                {(selectedAlert.local_explanation || []).map((f) => {
                  const metaForFeature = getFeatureMeta(f.feature)
                  const width = (safeNumber(f.abs_contribution, 0) / Math.max(1e-9, selectedMaxAbs)) * 100
                  const dir = String(f.direction || '')
                  return (
                    <li key={`${selectedAlert.id}-${f.feature}`}>
                      <div className="feature-row"><span className="feature-name">{metaForFeature.label}</span><span className="feature-val">{safeNumber(f.contribution).toFixed(4)}</span></div>
                      <div className="feature-meta">{f.feature} | {metaForFeature.description}</div>
                      <div className="feature-meta">{directionText(dir)}</div>
                      <div className="bar-wrap"><div className={`bar ${dir === 'benign' ? 'benign' : 'attack'}`} style={{ width: `${Math.min(100, width)}%` }} /></div>
                    </li>
                  )
                })}
              </ul>
            </>
          )}
        </div>
      </section>

      <section className="card panel">
        <h2>Global Explanations</h2>
        <div className="global-grid">
          <div>
            <h3>Overall Top Features</h3>
            <ul className="simple-list">
              {(globalExplanations?.overall_top_features || []).slice(0, 12).map((r) => {
                const m = getFeatureMeta(r.feature)
                return <li key={r.feature}><span>{m.label} <span className="feature-tech">({r.feature})</span></span><span>{safeNumber(r.score).toFixed(3)}</span></li>
              })}
            </ul>
          </div>
          <div>
            <h3>Per-Protocol Drivers</h3>
            <div className="protocol-columns">
              {(globalExplanations?.protocols || []).map((p) => (
                <div key={p.protocol} className="protocol-card">
                  <div className="protocol-head"><span className="badge">{p.protocol}</span><span>thr={safeNumber(p.threshold).toFixed(4)}</span></div>
                  <ul className="simple-list">
                    {(p.top_features || []).slice(0, 8).map((f) => {
                      const m = getFeatureMeta(f.feature)
                      return <li key={`${p.protocol}-${f.feature}`}><span>{m.label} <span className="feature-tech">({f.feature})</span></span><span>{safeNumber(f.mean_abs_contribution).toFixed(3)}</span></li>
                    })}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
