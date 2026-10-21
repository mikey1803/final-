import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { Dices, Lock, Crosshair, Repeat, CheckCircle2, XCircle } from 'lucide-react'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'

// Each method gets a genuinely different identity, not a shared card
// template: an icon and micro-copy that describe its actual mechanism, a
// color reserved for the two methods that learn (Thompson = adaptive
// indigo, LinUCB = a second "bound" teal, distinct from Thompson's), and
// the two that don't learn stay neutral gray, differentiated by icon and
// motion character rather than a borrowed hue. Only real fields from
// GET .../compare-live are used (method, chosen_strategy, outcome,
// learns) — no synthesized distributions.
const IDENTITY = {
  'Thompson Sampling': {
    icon: Dices,
    tone: 'var(--color-lav-primary)',
    tag: 'samples from belief',
    settle: true, // stochastic draw -> a brief settle wobble on reveal
  },
  'Fixed Rule (blind)': {
    icon: Lock,
    tone: 'var(--color-lav-muted)',
    tag: 'fixed lookup, never updates',
    settle: false, // deterministic -> reveals flat, no motion, on purpose
  },
  LinUCB: {
    icon: Crosshair,
    tone: 'var(--color-lav-teal)',
    tag: 'confidence-bound selection',
    settle: true,
  },
  'Naive Retry': {
    icon: Repeat,
    tone: 'var(--color-lav-muted)',
    tag: 'repeats blindly, no memory',
    settle: false,
  },
}
const FALLBACK_IDENTITY = { icon: Dices, tone: 'var(--color-lav-muted)', tag: '', settle: false }

const STAGE_ORDER = ['selecting', 'selected', 'executing', 'outcome']
const TIMING = { selected: 450, executing: 750, outcome: 1150 }

function MethodColumn({ r }) {
  const [stage, setStage] = useState('selecting')
  const id = IDENTITY[r.method] ?? FALLBACK_IDENTITY
  const Icon = id.icon

  useEffect(() => {
    setStage('selecting')
    const timers = Object.entries(TIMING).map(([s, ms]) => setTimeout(() => setStage(s), ms))
    return () => timers.forEach(clearTimeout)
  }, [r])

  const at = (s) => STAGE_ORDER.indexOf(stage) >= STAGE_ORDER.indexOf(s)
  const strategyLabel = RECOVERY_STRATEGIES.find((s) => s.id === r.chosen_strategy)?.label ?? r.chosen_strategy
  const success = r.outcome === 'success'

  return (
    <div className="flex flex-col gap-2.5 border-t-2 pt-3" style={{ borderColor: id.tone }}>
      <div className="flex items-center gap-2">
        <Icon size={14} style={{ color: id.tone }} className="shrink-0" />
        <div className="min-w-0">
          <p className="truncate font-sans text-xs font-semibold text-lav-text">{r.method}</p>
          <p className="font-mono text-[10px] text-lav-muted">{id.tag}</p>
        </div>
      </div>

      <div className="flex min-h-[2.2rem] items-center font-mono text-xs">
        {at('selected') ? (
          <motion.span
            initial={id.settle ? { scale: 1.15, opacity: 0 } : { opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={id.settle ? { type: 'spring', stiffness: 300, damping: 14 } : { duration: 0.15 }}
            style={{ color: id.tone }}
          >
            {strategyLabel}
          </motion.span>
        ) : (
          <span className="text-lav-muted">selecting...</span>
        )}
      </div>

      <div className="flex min-h-[1.4rem] items-center gap-1.5 font-sans text-xs">
        {at('outcome') ? (
          <span className={success ? 'flex items-center gap-1.5 font-medium text-lav-success' : 'flex items-center gap-1.5 font-medium text-lav-cost'}>
            {success ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
            {success ? 'Success' : 'Failed'}
          </span>
        ) : at('executing') ? (
          <span className="text-lav-muted">executing...</span>
        ) : null}
      </div>
    </div>
  )
}

/** Renders GET .../compare-live's {true_failure_type, results: [{method,
 * chosen_strategy, outcome, learns}]} as four columns that visually argue
 * their own mechanism, not four identical cards. */
export function CompareLivePanel({ compareResult }) {
  if (!compareResult) return null

  return (
    <div className="mt-2 grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
      {compareResult.results.map((r) => (
        <MethodColumn key={r.method} r={r} />
      ))}
    </div>
  )
}
