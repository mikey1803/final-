import { useCallback, useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Loader2, AlertTriangle, FlaskConical, TrendingUp } from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { ChartTooltip } from '../components/ChartTooltip'
import { InteractiveLegend } from '../components/InteractiveLegend'
import { useSeriesToggle } from '../hooks/useSeriesToggle'
import { colorFor } from '../lib/colors'
import { cn } from '../lib/utils'
import { FAILURE_TYPES } from '../data/failureTypes'
import { RECOVERY_STRATEGIES } from '../data/recoveryStrategies'
import { MATRIX_PRESETS } from '../data/matrixPresets'
import { checkHealth, whatIfMatrix } from '../lib/demoApi'

const SERIES = [
  { key: 'thompson_success_rate', name: 'Thompson Sampling', color: colorFor('Adaptive (Thompson)') },
  { key: 'fixed_rule_success_rate', name: 'Fixed Rule', color: colorFor('Fixed Rule') },
  { key: 'linucb_success_rate', name: 'LinUCB', color: colorFor('LinUCB') },
  { key: 'naive_retry_success_rate', name: 'Naive Retry', color: colorFor('Naive Retry') },
]

function cloneMatrix(rows) {
  const copy = {}
  for (const ft of Object.keys(rows)) copy[ft] = { ...rows[ft] }
  return copy
}

/** Editable 7x6 success-rate grid. Each cell is P(success | true failure
 * type, recovery strategy) — the exact hidden quantity every selector in
 * this project has to learn without seeing it. Editing a cell is the whole
 * point: it lets a visitor construct their own environment instead of only
 * ever seeing the canonical one. */
function MatrixGrid({ matrix, onChange }) {
  return (
    <div className="overflow-x-auto">
      <div
        className="grid gap-px overflow-hidden rounded-md border border-lav-border bg-lav-border"
        style={{ gridTemplateColumns: `160px repeat(${RECOVERY_STRATEGIES.length}, minmax(76px, 1fr))`, minWidth: 680 }}
      >
        <div className="bg-lav-bg px-2 py-2" />
        {RECOVERY_STRATEGIES.map((s) => (
          <div key={s.id} className="flex items-center justify-center bg-lav-bg px-1 py-2 text-center font-mono text-[10px] text-lav-muted">
            {s.label}
          </div>
        ))}
        {FAILURE_TYPES.map((ft) => (
          <div key={ft.id} className="contents">
            <div className="flex items-center gap-1.5 bg-lav-bg px-2 py-1.5 font-sans text-xs text-lav-text">
              <ft.icon size={12} style={{ color: ft.color }} className="shrink-0" />
              <span className="truncate">{ft.label}</span>
            </div>
            {RECOVERY_STRATEGIES.map((s) => (
              <input
                key={s.id}
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={matrix[ft.id]?.[s.id] ?? 0}
                onChange={(e) => {
                  const raw = e.target.value
                  const v = raw === '' ? 0 : Math.max(0, Math.min(1, Number(raw)))
                  onChange(ft.id, s.id, v)
                }}
                className="bg-lav-surface px-1.5 py-1.5 text-center font-mono text-[11px] text-lav-text focus:bg-lav-surface-raised focus:outline-none"
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/** Lets a visitor construct their own (failure_type, strategy) success-rate
 * matrix and race the real selectors against it — turns the paper's Task 9
 * finding ("2 of 3 hand-constructed alternative matrices break the
 * 'Fixed Rule wins' ranking outright") into something you test yourself
 * instead of only reading about. */
export function MatrixExplorer() {
  const [backendUp, setBackendUp] = useState(null)
  const [presetId, setPresetId] = useState('canonical')
  const [matrix, setMatrix] = useState(() => cloneMatrix(MATRIX_PRESETS[0].rows))
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const series = useSeriesToggle()
  const dim = (key) => (series.hoveredKey && series.hoveredKey !== key ? 0.25 : 1)

  useEffect(() => {
    checkHealth().then(() => setBackendUp(true)).catch(() => setBackendUp(false))
  }, [])

  const applyPreset = (id) => {
    const preset = MATRIX_PRESETS.find((p) => p.id === id)
    if (!preset) return
    setPresetId(id)
    setMatrix(cloneMatrix(preset.rows))
    setResult(null)
  }

  const handleCellChange = (ftId, sId, value) => {
    setMatrix((prev) => ({ ...prev, [ftId]: { ...prev[ftId], [sId]: value } }))
  }

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await whatIfMatrix(matrix, 200)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }, [matrix])

  const activePreset = MATRIX_PRESETS.find((p) => p.id === presetId)

  return (
    <div>
      <PageHeader
        title="Try Your Own Scenario"
        subtitle="Every method in this project learns (or is designed against) a hidden success-rate matrix it never gets to see directly. Edit it yourself, or load one of three real alternative matrices from the paper's Task 9 robustness test, and race Thompson Sampling, Fixed Rule, LinUCB, and Naive Retry against it — 200 fresh episodes, computed live."
      />

      {backendUp === false && (
        <div className="flex items-start gap-3 border-t border-lav-cost/40 pt-4">
          <AlertTriangle className="mt-0.5 shrink-0 text-lav-cost" size={18} />
          <div>
            <p className="font-sans text-sm font-semibold text-lav-text">Backend not reachable</p>
            <p className="mt-1 font-sans text-sm text-lav-muted">
              This tab calls a real FastAPI server at <code className="font-mono text-lav-text">localhost:8000</code>. Start it first:
            </p>
            <pre className="mt-2 overflow-x-auto rounded-md border border-lav-border bg-lav-surface px-3 py-2 font-mono text-xs text-lav-text">
              cd backend{'\n'}uvicorn api.main:app --reload --port 8000
            </pre>
          </div>
        </div>
      )}

      {backendUp !== false && (
        <>
          <div className="border-t border-lav-border py-4">
            <p className="mb-2.5 font-mono text-[10px] uppercase tracking-wider text-lav-muted">Load a scenario</p>
            <div className="flex flex-wrap gap-2">
              {MATRIX_PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => applyPreset(p.id)}
                  className={cn(
                    'rounded-md border px-3 py-1.5 text-left font-sans text-xs transition-colors',
                    presetId === p.id
                      ? 'border-lav-primary/50 bg-lav-primary/10 text-lav-text'
                      : 'border-lav-border bg-lav-surface text-lav-muted hover:text-lav-text',
                  )}
                >
                  {p.label}
                </button>
              ))}
            </div>
            {activePreset && <p className="mt-2 font-sans text-xs text-lav-muted">{activePreset.description}</p>}
          </div>

          <div className="border-t border-lav-border py-4">
            <p className="mb-2.5 font-mono text-[10px] uppercase tracking-wider text-lav-muted">
              Success-rate matrix (edit any cell)
            </p>
            <MatrixGrid matrix={matrix} onChange={handleCellChange} />
          </div>

          <div className="border-t border-lav-border py-4">
            <motion.button
              onClick={handleRun}
              disabled={loading}
              whileTap={loading ? undefined : { scale: 0.97 }}
              transition={{ type: 'spring', bounce: 0, duration: 0.25 }}
              className="flex items-center gap-2 rounded bg-lav-primary px-4 py-1.5 font-sans text-sm font-medium text-white disabled:opacity-40"
            >
              {loading ? <Loader2 className="animate-spin" size={14} /> : <FlaskConical size={14} />}
              {loading ? 'Running 200 episodes...' : 'Run comparison'}
            </motion.button>
          </div>

          {error && <div className="border-t border-lav-cost/40 py-3 font-sans text-sm text-lav-cost">{error}</div>}

          {result && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="border-t border-lav-border py-5">
              <div
                className={cn(
                  'mb-4 flex items-center gap-2 border-l-2 py-1.5 pl-3 font-sans text-sm',
                  result.fixed_rule_wins ? 'border-lav-success text-lav-success' : 'border-lav-amber text-lav-amber',
                )}
              >
                <TrendingUp size={14} className="shrink-0" />
                {result.fixed_rule_wins
                  ? 'Fixed Rule still wins in this scenario — the headline ranking holds.'
                  : `Fixed Rule no longer wins here. Final ranking: ${result.final_ranking.map((r) => r.method).join(' > ')}.`}
              </div>

              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={result.rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                  <XAxis
                    dataKey="episode"
                    tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: 'var(--color-lav-muted)' }}
                    stroke="var(--color-lav-border)"
                  />
                  <YAxis
                    domain={[0, 1]}
                    allowDataOverflow
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                    tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: 'var(--color-lav-muted)' }}
                    stroke="var(--color-lav-border)"
                    width={40}
                  />
                  <Tooltip
                    content={<ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} labelFormatter={(l) => `episode ${l}`} />}
                    cursor={{ stroke: 'var(--color-lav-primary)', strokeWidth: 1, strokeDasharray: '4 4' }}
                  />
                  <Legend content={<InteractiveLegend hidden={series.hidden} onToggle={series.toggle} onHover={series.setHovered} />} />
                  {SERIES.map((s) => (
                    <Line
                      key={s.key}
                      type="monotone"
                      dataKey={s.key}
                      name={s.name}
                      stroke={s.color}
                      strokeWidth={2.2}
                      strokeOpacity={dim(s.key)}
                      dot={false}
                      hide={series.isHidden(s.key)}
                      animationDuration={400}
                      animationEasing="ease-out"
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>

              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {result.final_ranking.map((r, i) => (
                  <div key={r.method} className="rounded-md border border-lav-border bg-lav-surface px-3 py-2">
                    <p className="font-mono text-[10px] text-lav-muted">#{i + 1}</p>
                    <p className="font-sans text-xs font-medium text-lav-text">{r.method}</p>
                    <p className="font-mono text-sm text-lav-primary">{(r.success_rate * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </>
      )}
    </div>
  )
}
