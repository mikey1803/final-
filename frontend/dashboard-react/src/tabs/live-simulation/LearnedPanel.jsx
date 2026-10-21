import { useState } from 'react'
import { motion } from 'motion/react'
import { CheckCircle2, Circle, Info } from 'lucide-react'
import { FAILURE_TYPES } from '../../data/failureTypes'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'
import { explainFailureType } from '../../lib/demoApi'

/** Reads straight off GET /api/demo/state's shape:
 * { episode_number, best_strategy_per_failure_type: { [ft]: {strategy, confidence} } } */
export function LearnedPanel({ demoState }) {
  const [openId, setOpenId] = useState(null)
  const [explainText, setExplainText] = useState('')
  const [explainLoading, setExplainLoading] = useState(false)

  const toggleExplain = async (failureTypeId) => {
    if (openId === failureTypeId) {
      setOpenId(null)
      return
    }
    setOpenId(failureTypeId)
    setExplainLoading(true)
    setExplainText('')
    try {
      const data = await explainFailureType(failureTypeId)
      setExplainText(data.explanation)
    } catch (e) {
      setExplainText(e.message || 'Could not load explanation.')
    } finally {
      setExplainLoading(false)
    }
  }

  if (!demoState) return null
  const best = demoState.best_strategy_per_failure_type
  const rows = FAILURE_TYPES.map((f) => {
    const entry = best[f.id]
    // "seen" must check every strategy's confidence, not just the best one —
    // a type that has only ever failed so far pushes arms *below* 0.5, which
    // would leave the argmax sitting on an untouched 0.5 arm and wrongly
    // read as never-seen if we only looked at the best value.
    const rowConfidences = demoState.confidences?.[f.id] ?? {}
    const seen = Object.values(rowConfidences).some((v) => v !== 0.5)
    return { failureType: f, strategy: entry?.strategy, confidence: entry?.confidence ?? 0.5, seen }
  })
  const seenCount = rows.filter((r) => r.seen).length

  return (
    <div className="border-t border-lav-border pt-3">
      <div className="mb-2.5 flex items-center justify-between">
        <p className="font-mono text-[10px] uppercase tracking-wider text-lav-muted">
          Learned so far ({seenCount}/7)
        </p>
        <span className="font-mono text-[10px] text-lav-muted">ep. {demoState.episode_number}</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {rows.map((r) => {
          const seen = r.seen
          const strategy = RECOVERY_STRATEGIES.find((s) => s.id === r.strategy)
          const isOpen = openId === r.failureType.id
          return (
            <div key={r.failureType.id} className="rounded-lg">
              <div
                className={`flex items-center gap-2 px-2 py-2 font-sans text-sm transition-opacity duration-300 ${seen ? 'opacity-100' : 'opacity-35'}`}
              >
                {seen ? (
                  <CheckCircle2 size={15} className="shrink-0 text-lav-success" />
                ) : (
                  <Circle size={15} className="shrink-0 text-lav-muted" />
                )}
                <span className="min-w-0 flex-1 truncate text-lav-text">{r.failureType.label}</span>
                <span className="text-lav-border">&rarr;</span>
                {seen ? (
                  <motion.span
                    key={r.strategy}
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="font-mono font-medium text-lav-primary"
                  >
                    {strategy?.label ?? r.strategy}
                  </motion.span>
                ) : (
                  <span className="font-mono font-medium text-lav-muted">?</span>
                )}
                <button
                  onClick={() => toggleExplain(r.failureType.id)}
                  className="shrink-0 text-lav-muted transition-colors hover:text-lav-primary"
                  title="Why this strategy?"
                >
                  <Info size={13} />
                </button>
              </div>
              {isOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="overflow-hidden px-2 pb-2"
                >
                  <p className="rounded-md bg-lav-bg px-2.5 py-2 font-sans text-xs leading-relaxed text-lav-muted">
                    {explainLoading ? 'Loading...' : explainText}
                  </p>
                </motion.div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
