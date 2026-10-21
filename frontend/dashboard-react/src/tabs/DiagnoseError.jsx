import { useCallback, useState } from 'react'
import { motion } from 'motion/react'
import { Loader2, AlertTriangle, Search, ShieldCheck } from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { cn } from '../lib/utils'
import { FAILURE_TYPES } from '../data/failureTypes'
import { RECOVERY_STRATEGIES } from '../data/recoveryStrategies'
import { diagnose } from '../lib/demoApi'

const EXAMPLE_ERROR = 'requests.exceptions.Timeout: HTTPSConnectionPool(host=\'api.weather.com\', port=443): Read timed out after 30s'

/** One failure type's probability, rendered as a filled bar against the
 * type's own fixed identity color (same convention as FailureEventGrid /
 * StrategyBeliefBars) so the reader can see the classifier's full belief,
 * not just the single winning label. */
function ProbabilityRow({ failureType, probability, isTop }) {
  return (
    <div className="flex items-center gap-3">
      <failureType.icon size={13} className="shrink-0" style={{ color: failureType.color }} />
      <span className={cn('w-40 shrink-0 truncate font-sans text-xs', isTop ? 'font-semibold text-lav-text' : 'text-lav-muted')}>
        {failureType.label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-lav-border">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.round(probability * 100)}%` }}
          transition={{ type: 'spring', bounce: 0, duration: 0.5 }}
          className="h-full rounded-full"
          style={{ backgroundColor: failureType.color }}
        />
      </div>
      <span className="w-12 shrink-0 text-right font-mono text-xs text-lav-muted">{Math.round(probability * 100)}%</span>
    </div>
  )
}

/** Paste any real error/log text and get the actual trained classifier's
 * prediction plus the actual bandit's current best-recommended recovery
 * strategy for it. Unlike the rest of the live demo, this never mutates
 * anything — there's no ground truth for text pasted from outside the
 * demo, so POST /api/demo/diagnose is read-only by construction. */
export function DiagnoseError() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const handleDiagnose = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await diagnose(text.trim())
      setResult(data)
    } catch (e) {
      setError(e.message || 'Diagnosis failed')
    } finally {
      setLoading(false)
    }
  }, [text])

  const predictedType = result ? FAILURE_TYPES.find((f) => f.id === result.predicted_failure_type) : null
  const recommendedStrategy = result ? RECOVERY_STRATEGIES.find((s) => s.id === result.recommended_strategy) : null
  const rankedProbabilities = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div>
      <PageHeader
        title="Diagnose an Error"
        subtitle="Paste any real error or log text — your own agent's, anyone's. The actual trained classifier (TF-IDF + LinearSVC) predicts its failure type, and the actual persistent bandit recommends its current best recovery strategy. Read-only: nothing here feeds back into the demo's learned state."
      />

      <div className="border-t border-lav-border py-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={`Paste an error message or log line here, e.g.:\n${EXAMPLE_ERROR}`}
          rows={5}
          className="w-full resize-y rounded-md border border-lav-border bg-lav-surface px-3 py-2.5 font-mono text-sm text-lav-text placeholder:text-lav-muted/60 focus:border-lav-primary/50 focus:outline-none"
        />
        <div className="mt-3 flex items-center gap-3">
          <motion.button
            onClick={handleDiagnose}
            disabled={loading || !text.trim()}
            whileTap={loading ? undefined : { scale: 0.97 }}
            transition={{ type: 'spring', bounce: 0, duration: 0.25 }}
            className="flex items-center gap-2 rounded bg-lav-primary px-4 py-1.5 font-sans text-sm font-medium text-white disabled:opacity-40"
          >
            {loading ? <Loader2 className="animate-spin" size={14} /> : <Search size={14} />}
            {loading ? 'Diagnosing...' : 'Diagnose'}
          </motion.button>
          <button
            onClick={() => setText(EXAMPLE_ERROR)}
            disabled={loading}
            className="font-sans text-xs text-lav-muted transition-colors hover:text-lav-text disabled:opacity-40"
          >
            Try an example
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-3 border-t border-lav-cost/40 py-4">
          <AlertTriangle className="mt-0.5 shrink-0 text-lav-cost" size={18} />
          <p className="font-sans text-sm text-lav-cost">{error}</p>
        </div>
      )}

      {result && !error && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="border-t border-lav-border py-5">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div>
              <p className="mb-2.5 font-mono text-[10px] uppercase tracking-wider text-lav-muted">Predicted failure type</p>
              {predictedType && (
                <div className="mb-4 flex items-center gap-2.5 rounded-md border border-lav-border bg-lav-surface px-3 py-2.5">
                  <predictedType.icon size={16} style={{ color: predictedType.color }} className="shrink-0" />
                  <div>
                    <p className="font-sans text-sm font-medium text-lav-text">{predictedType.label}</p>
                    <p className="font-mono text-xs text-lav-amber">
                      {Math.round((result.probabilities[predictedType.id] ?? 0) * 100)}% confidence
                    </p>
                  </div>
                </div>
              )}
              <div className="flex flex-col gap-2">
                {rankedProbabilities.map(([id, p]) => {
                  const ft = FAILURE_TYPES.find((f) => f.id === id)
                  if (!ft) return null
                  return <ProbabilityRow key={id} failureType={ft} probability={p} isTop={id === result.predicted_failure_type} />
                })}
              </div>
            </div>

            <div>
              <p className="mb-2.5 font-mono text-[10px] uppercase tracking-wider text-lav-muted">Recommended recovery strategy</p>
              <div className="rounded-md border border-lav-border bg-lav-surface px-4 py-3.5">
                <p className="font-mono text-lg font-semibold text-lav-primary">{recommendedStrategy?.label ?? result.recommended_strategy}</p>
                <p className="mt-1 font-sans text-xs text-lav-muted">{recommendedStrategy?.description}</p>
                <p className="mt-2 font-mono text-xs text-lav-success">
                  {Math.round(result.recommended_strategy_confidence * 100)}% current belief of success
                </p>
              </div>
              <p className="mt-4 flex items-start gap-2 font-sans text-xs leading-relaxed text-lav-muted">
                <ShieldCheck size={14} className="mt-0.5 shrink-0 text-lav-muted" />
                Diagnosis only — this reads the bandit's current learned state but does not update it, and doesn't
                consume an episode. Nothing about this request changes what Live Simulation or Beat the Bandit see.
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
