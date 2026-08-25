import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { CheckCircle2, XCircle } from 'lucide-react'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'
import { PipelineLayer, PipelineConnector } from './PipelineLayer'

const STAGE_ORDER = ['selecting', 'selected', 'executing', 'outcome']
const TIMING = { selected: 600, executing: 900, outcome: 1400 }

function MethodColumn({ r }) {
  const [stage, setStage] = useState('selecting')

  useEffect(() => {
    setStage('selecting')
    const timers = Object.entries(TIMING).map(([s, ms]) => setTimeout(() => setStage(s), ms))
    return () => timers.forEach(clearTimeout)
  }, [r])

  const at = (s) => STAGE_ORDER.indexOf(stage) >= STAGE_ORDER.indexOf(s)
  const strategyLabel = RECOVERY_STRATEGIES.find((s) => s.id === r.chosen_strategy)?.label ?? r.chosen_strategy
  const success = r.outcome === 'success'

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-stretch">
      <div className="rounded-lg border border-lav-border bg-lav-bg/60 px-3 py-2 text-center">
        <p className="truncate font-sans text-xs font-semibold text-lav-text">{r.method}</p>
        <p className="font-sans text-[10px] text-lav-muted">{r.learns ? 'learns over time' : 'no learning'}</p>
      </div>

      <PipelineConnector />
      <PipelineLayer number="A" name="Select Strategy" processing={stage === 'selecting'}>
        {at('selected') ? (
          <p className="font-mono text-sm font-medium text-lav-primary">{strategyLabel}</p>
        ) : (
          <p className="font-mono text-xs text-lav-muted">Choosing...</p>
        )}
      </PipelineLayer>

      {at('executing') && (
        <>
          <PipelineConnector />
          <PipelineLayer number="B" name="Execute" processing={stage === 'executing'}>
            {at('outcome') ? (
              <div
                className={`flex items-center gap-1.5 font-sans text-sm font-semibold ${
                  success ? 'text-lav-success' : 'text-lav-cost'
                }`}
              >
                {success ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                {success ? 'Success' : 'Failed'}
              </div>
            ) : (
              <p className="font-mono text-xs text-lav-muted">Executing {strategyLabel}...</p>
            )}
          </PipelineLayer>
        </>
      )}
    </motion.div>
  )
}

/** Renders GET .../compare-live's {true_failure_type, results: [{method, chosen_strategy, outcome, learns}]}
 * as parallel columns, each running its own 2-stage mini pipeline reveal. */
export function CompareLivePanel({ compareResult }) {
  if (!compareResult) return null

  return (
    <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {compareResult.results.map((r) => (
        <MethodColumn key={r.method} r={r} />
      ))}
    </div>
  )
}
