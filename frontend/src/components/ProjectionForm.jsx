import { useState } from 'react'
import { api } from '../api.js'

export default function ProjectionForm({ playerId, week, season }) {
  const [opponent, setOpponent] = useState('GB')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function run(refresh) {
    if (!playerId) return
    setBusy(true); setError(null)
    const started = performance.now()
    try {
      const body = await api.predict({ player_id: playerId, week, season, opponent, is_home: true })
      setResult({ ...body, elapsed_ms: Math.round(performance.now() - started), refresh })
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="controls">
        <input
          value={opponent}
          onChange={(e) => setOpponent(e.target.value.toUpperCase().slice(0, 4))}
          placeholder="Opponent"
          style={{ width: 100 }}
        />
        <button className="primary" disabled={!playerId || busy} onClick={() => run(false)}>
          {busy ? 'Projecting…' : 'Project'}
        </button>
      </div>
      {error && <p className="err">{error}</p>}
      {result && (
        <>
          <div className="kv"><span>Projected points</span><span className="pts">{result.prediction.toFixed(1)}</span></div>
          <div className="kv"><span>Confidence</span><span>{result.confidence?.toFixed(2) ?? '—'}</span></div>
          <div className="kv"><span>Model</span><span><span className="badge">{result.model_version}</span></span></div>
          <div className="kv">
            <span>Served from</span>
            <span>{result.source} · {result.elapsed_ms} ms</span>
          </div>
          <p className="empty" style={{ marginTop: 8 }}>
            Repeat the same request to see the cache path — <code>source: cache</code> skips
            inference entirely.
          </p>
        </>
      )}
      {!playerId && <p className="empty">Pick a player from the leaderboard first.</p>}
    </>
  )
}
