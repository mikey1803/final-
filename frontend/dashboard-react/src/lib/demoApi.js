// Client for the live-demo FastAPI backend (backend/api/main.py). Every
// call here hits a real HTTP server running the real trained classifier and
// real persistent Thompson Sampling selector — there is no local/synthetic
// fallback data path in this module.
const API_BASE = 'http://localhost:8000'

async function request(path, options) {
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, options)
  } catch {
    throw new Error(
      'Cannot reach the demo backend at localhost:8000. Start it with: uvicorn api.main:app --port 8000 (from backend/)',
    )
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail || `Backend returned ${res.status}`)
  }
  return res.json()
}

const post = (path, body) =>
  request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })

export const checkHealth = () => request('/api/demo/health')
export const getDemoState = () => request('/api/demo/state')
export const injectManual = (failureType) => post('/api/demo/inject-manual', { failure_type: failureType })
export const injectNatural = (userRequest) => post('/api/demo/inject-natural', { user_request: userRequest })
export const resetDemo = () => post('/api/demo/reset')
export const compareLive = (failureType) => post('/api/demo/compare-live', { failure_type: failureType })
export const triggerShift = () => post('/api/demo/trigger-shift')
export const batchRun = (nEpisodes) => post('/api/demo/batch-run', { n_episodes: nEpisodes })
export const explainFailureType = (failureType) => request(`/api/demo/explain/${failureType}`)
export const retryRecovery = (trueFailureType, predictedFailureType, attemptNumber) =>
  post('/api/demo/retry', {
    true_failure_type: trueFailureType,
    predicted_failure_type: predictedFailureType,
    attempt_number: attemptNumber,
  })
