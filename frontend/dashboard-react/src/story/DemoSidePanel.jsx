import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle2, Circle } from 'lucide-react'
import { Card } from '../components/ui/Card'
import { FAILURE_TYPES } from '../data/failureTypes'
import { RECOVERY_STRATEGIES } from '../data/recoveryStrategies'
import { FAILURE_LEARNED_MAPPING } from './data/failureLearnedMapping'

function AnimatedStat({ value }) {
  return (
    <AnimatePresence mode="wait">
      <motion.span
        key={value}
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 6 }}
        transition={{ duration: 0.18 }}
        className="font-mono text-3xl font-semibold"
        style={{ color: 'var(--color-lav-live)' }}
      >
        {value}
      </motion.span>
    </AnimatePresence>
  )
}

export function DemoSidePanel({ episodesSoFar, windowSuccessRate, cumulativeSuccessRate }) {
  const learnedCount = FAILURE_LEARNED_MAPPING.filter((m) => m.revealEpisode <= episodesSoFar).length

  return (
    <div className="flex flex-col gap-4">
      <Card hover={false} className="p-5">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="font-sans text-xs text-lav-muted">Success rate (this window)</p>
            <AnimatedStat value={`${(windowSuccessRate * 100).toFixed(1)}%`} />
          </div>
          <div>
            <p className="font-sans text-xs text-lav-muted">Cumulative success rate</p>
            <AnimatedStat value={`${(cumulativeSuccessRate * 100).toFixed(1)}%`} />
          </div>
        </div>
        <div className="mt-4 border-t border-lav-border pt-3">
          <p className="font-sans text-xs text-lav-muted">Episodes simulated so far</p>
          <AnimatedStat value={episodesSoFar} />
          <span className="ml-1 font-sans text-xs text-lav-muted">/ 500</span>
        </div>
      </Card>

      <Card hover={false} className="p-5" style={{ borderColor: 'rgba(52,211,153,0.25)', backgroundColor: 'rgba(52,211,153,0.03)' }}>
        <p className="mb-3 font-sans text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--color-lav-live)' }}>
          What it has learned so far ({learnedCount}/7)
        </p>
        <div className="flex flex-col gap-1.5">
          {FAILURE_LEARNED_MAPPING.map((m) => {
            const failureType = FAILURE_TYPES.find((f) => f.id === m.failureTypeId)
            const strategy = RECOVERY_STRATEGIES.find((s) => s.id === m.strategy)
            const learned = m.revealEpisode <= episodesSoFar
            return (
              <div
                key={m.failureTypeId}
                className={`flex items-center gap-2 rounded-lg px-2 py-2 font-sans text-sm transition-opacity duration-300 ${learned ? 'opacity-100' : 'opacity-35'}`}
              >
                {learned ? (
                  <CheckCircle2 size={15} className="shrink-0 text-lav-success" />
                ) : (
                  <Circle size={15} className="shrink-0 text-lav-muted" />
                )}
                <span className="text-lav-text">{failureType.label}</span>
                <span className="text-lav-border">&rarr;</span>
                <span className="font-mono font-medium" style={{ color: 'var(--color-lav-live)' }}>
                  {learned ? strategy.label : '?'}
                </span>
              </div>
            )
          })}
        </div>
      </Card>
    </div>
  )
}
