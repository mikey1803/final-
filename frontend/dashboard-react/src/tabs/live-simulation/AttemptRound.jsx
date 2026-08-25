import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { TraceNode, TraceConnector } from './TraceNode'
import { BeliefDelta } from './BeliefDelta'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'

const STAGE_ORDER = ['l2-processing', 'l2-bars', 'l2-selected', 'l3-processing', 'l3-outcome', 'l4-processing', 'l4-revealed']

const TIMING = {
  'l2-bars': 600,
  'l2-selected': 1400,
  'l3-processing': 1700,
  'l3-outcome': 2200,
  'l4-processing': 2500,
  'l4-revealed': 3000,
}

const DONE_DELAY = TIMING['l4-revealed'] + 400

/** One round of strategy-select -> execute -> belief-update for a single
 * attempt, rendered as two hero-trace nodes ("Strategy chosen" / "Outcome").
 * `data` is already-fetched (the real API call already happened); this only
 * stages its reveal. Calls onDone(success) once the round has fully
 * revealed, so the parent can decide whether to retry.
 *
 * `isPrimary` (attempt 1) renders inline as nodes 4-5 of the main hero row
 * (via `display: contents`, so it participates directly in the parent flex
 * row without an extra wrapping box); retries (isPrimary=false) render as a
 * smaller standalone row stacked below, with their own attempt-number tag
 * and connector — same timing/logic, only the visual scale differs. */
export function AttemptRound({ attemptNumber, data, isPrimary = false, onDone }) {
  const [stage, setStage] = useState('l2-processing')

  useEffect(() => {
    setStage('l2-processing')
    const timers = Object.entries(TIMING).map(([s, ms]) => setTimeout(() => setStage(s), ms))
    const doneTimer = setTimeout(() => onDone(data.outcome === 'success'), DONE_DELAY)
    return () => {
      timers.forEach(clearTimeout)
      clearTimeout(doneTimer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  const at = (s) => STAGE_ORDER.indexOf(stage) >= STAGE_ORDER.indexOf(s)
  const chosenBefore = data.strategy_scores_before?.[data.chosen_strategy] ?? 0.5
  const chosenAfter = data.strategy_scores_after?.[data.chosen_strategy] ?? 0.5
  const chosenLabel = RECOVERY_STRATEGIES.find((s) => s.id === data.chosen_strategy)?.label ?? data.chosen_strategy
  const success = data.outcome === 'success'
  const size = isPrimary ? 'lg' : 'sm'

  const strategyNode = (
    <TraceNode index={isPrimary ? 4 : 'A'} label="Strategy chosen" size={size} status={at('l2-selected') ? 'done' : 'active'}>
      {at('l2-selected') ? (
        <>
          {chosenLabel}
          <span className="block text-lav-muted">&mu;={Math.round(chosenBefore * 100)}%</span>
        </>
      ) : (
        'sampling...'
      )}
    </TraceNode>
  )

  const outcomeNode = (
    <TraceNode
      index={isPrimary ? 5 : 'B'}
      label="Outcome"
      size={size}
      status={!at('l3-processing') ? 'pending' : !at('l3-outcome') ? 'active' : success ? 'success' : 'fail'}
    >
      {at('l3-outcome') && (
        <>
          <span className={success ? 'text-lav-success' : 'text-lav-cost'}>{success ? 'Success' : 'Failed'}</span>
          {at('l4-revealed') && (
            <div className="mt-1 w-32">
              <BeliefDelta from={chosenBefore} to={chosenAfter} />
            </div>
          )}
        </>
      )}
    </TraceNode>
  )

  if (isPrimary) {
    return (
      <div className="contents">
        <TraceConnector />
        {strategyNode}
        {at('l3-processing') && (
          <>
            <TraceConnector />
            {outcomeNode}
          </>
        )}
      </div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="mt-4 flex flex-col items-center gap-2">
      <span className="rounded-full bg-lav-primary/15 px-3 py-1 font-mono text-[11px] font-semibold text-lav-primary">
        Attempt {attemptNumber}
      </span>
      <div className="flex flex-col items-center gap-2 md:flex-row md:items-start md:justify-center">
        {strategyNode}
        {at('l3-processing') && (
          <>
            <TraceConnector size="sm" />
            {outcomeNode}
          </>
        )}
      </div>
    </motion.div>
  )
}
