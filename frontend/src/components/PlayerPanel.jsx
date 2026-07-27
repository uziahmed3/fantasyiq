import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function PlayerPanel({ player, stats }) {
  if (!player) return <p className="empty">Select a player to see their game log.</p>

  const chart = [...(stats || [])]
    .sort((a, b) => a.week - b.week)
    .map((s) => ({ week: `W${s.week}`, points: s.fantasy_points }))

  return (
    <>
      <div className="kv"><span>Name</span><span>{player.name}</span></div>
      <div className="kv"><span>Team / Position</span><span>{player.team || '—'} · {player.position}</span></div>
      <div className="kv"><span>Age</span><span>{player.age ?? '—'}</span></div>
      {chart.length > 0 && (
        <div style={{ height: 170, marginTop: 14 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chart} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
              <CartesianGrid stroke="#262d36" vertical={false} />
              <XAxis dataKey="week" stroke="#8b949e" fontSize={11} />
              <YAxis stroke="#8b949e" fontSize={11} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #262d36', fontSize: 12 }}
              />
              <Line type="monotone" dataKey="points" stroke="#4f9cf9" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <table style={{ marginTop: 10 }}>
        <thead>
          <tr><th>Wk</th><th>Opp</th><th className="num">Tgt</th><th className="num">Rec</th><th className="num">Yds</th><th className="num">TD</th><th className="num">Pts</th></tr>
        </thead>
        <tbody>
          {(stats || []).map((s) => (
            <tr key={`${s.season}-${s.week}`}>
              <td>{s.week}</td>
              <td>{s.opponent || '—'}</td>
              <td className="num">{s.targets}</td>
              <td className="num">{s.receptions}</td>
              <td className="num">{s.yards.toFixed(0)}</td>
              <td className="num">{s.touchdowns}</td>
              <td className="pts">{s.fantasy_points.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
