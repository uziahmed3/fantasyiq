const BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  rankings: ({ week, season = 2023, position = 'WR', limit = 25 }) =>
    request(`/rankings?week=${week}&season=${season}&position=${position}&limit=${limit}`),
  players: ({ position = 'WR', name = '', limit = 50 }) =>
    request(`/players?position=${position}&name=${encodeURIComponent(name)}&limit=${limit}`),
  player: (id) => request(`/players/${id}`),
  stats: (id, season = 2023) => request(`/players/${id}/stats?season=${season}&limit=20`),
  predict: (body) => request('/predict', { method: 'POST', body: JSON.stringify(body) }),
  compare: (bodies) => request('/compare', { method: 'POST', body: JSON.stringify(bodies) }),
}
