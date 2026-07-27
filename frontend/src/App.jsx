import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import PlayerPanel from './components/PlayerPanel.jsx'
import ProjectionForm from './components/ProjectionForm.jsx'
import RankingsTable from './components/RankingsTable.jsx'

const POSITIONS = ['WR', 'RB', 'TE', 'QB']
const SEASON = 2023

export default function App() {
  const [week, setWeek] = useState(5)
  const [position, setPosition] = useState('WR')
  const [rankings, setRankings] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [player, setPlayer] = useState(null)
  const [stats, setStats] = useState([])
  const [error, setError] = useState(null)

  const loadRankings = useCallback(async () => {
    setError(null)
    try {
      setRankings(await api.rankings({ week, season: SEASON, position }))
    } catch (e) {
      setError(e.message)
      setRankings(null)
    }
  }, [week, position])

  useEffect(() => { loadRankings() }, [loadRankings])

  useEffect(() => {
    if (!selectedId) return
    let cancelled = false
    ;(async () => {
      try {
        const [p, s] = await Promise.all([api.player(selectedId), api.stats(selectedId, SEASON)])
        if (!cancelled) { setPlayer(p); setStats(s) }
      } catch (e) {
        if (!cancelled) setError(e.message)
      }
    })()
    return () => { cancelled = true }
  }, [selectedId])

  return (
    <div className="wrap">
      <header className="top">
        <h1>FantasyIQ</h1>
        <span className="tag">NFL fantasy point projections</span>
      </header>
      <p className="sub">
        FastAPI + Postgres + a separately deployed XGBoost/PyTorch inference service, behind a
        Redis cache-aside layer.
      </p>

      <div className="controls">
        <select value={position} onChange={(e) => setPosition(e.target.value)}>
          {POSITIONS.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={week} onChange={(e) => setWeek(Number(e.target.value))}>
          {Array.from({ length: 18 }, (_, i) => i + 1).map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
        <button onClick={loadRankings}>Refresh</button>
      </div>

      {error && <p className="err">{error}</p>}

      <div className="panel">
        <h2>Projected leaderboard {rankings ? `· ${rankings.model_version}` : ''}</h2>
        <RankingsTable data={rankings} onSelect={setSelectedId} />
      </div>

      <div className="grid2">
        <div className="panel">
          <h2>Game log</h2>
          <PlayerPanel player={player} stats={stats} />
        </div>
        <div className="panel">
          <h2>On-demand projection</h2>
          <ProjectionForm playerId={selectedId} week={week} season={SEASON} />
        </div>
      </div>

      <footer className="src">
        Data: nfl_data_py weekly stats. Projections are model output, not advice.
      </footer>
    </div>
  )
}
