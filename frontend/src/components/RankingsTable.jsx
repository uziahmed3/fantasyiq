export default function RankingsTable({ data, onSelect }) {
  if (!data?.rankings?.length) {
    return (
      <p className="empty">
        No projections stored for this week yet. Run <code>make ingest</code> to populate
        them, or use the projection panel to generate one on demand.
      </p>
    )
  }
  return (
    <table>
      <thead>
        <tr>
          <th className="rank">#</th>
          <th>Player</th>
          <th>Team</th>
          <th className="num">Confidence</th>
          <th className="num">Projected</th>
        </tr>
      </thead>
      <tbody>
        {data.rankings.map((row) => (
          <tr key={row.player_id} className="clickable" onClick={() => onSelect(row.player_id)}>
            <td className="rank">{row.rank}</td>
            <td>{row.name}</td>
            <td>{row.team || '—'}</td>
            <td className="num">{row.confidence ? row.confidence.toFixed(2) : '—'}</td>
            <td className="pts">{row.projected_points.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
