import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { CheckCircle2, UserCheck, RotateCw } from 'lucide-react'
import { EventRow, EventSpine } from './EventRow'
import { StrategyBeliefBars } from './StrategyBeliefBars'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'
import { FAILURE_TYPES } from '../../data/failureTypes'
import { retryRecovery } from '../../lib/demoApi'

const MAX_ATTEMPTS = 3
const CLASSIFY_MS = 550
const SELECT_MS = 500
const EXECUTE_MS = 550
const RETRY_LABEL_MS = 650

function attempt1FromResult(result) {
  return {
    strategy_scores_before: result.strategy_scores_before,
    strategy_scores_after: result.strategy_scores_after,
    chosen_strategy: result.chosen_strategy,
    outcome: result.outcome,
  }
}

/** One attempt's two rows (strategy selection, execution outcome), staged
 * the same way the old AttemptRound did — the API call already happened
 * (`data` is real, already-fetched), this only paces its reveal so the
 * user can actually follow what changed. Calls onDone(success) once fully
 * revealed. */
function AttemptEvents({ attemptNumber, data, onDone }) {
  const [stage, setStage] = useState('selecting') // selecting -> selected -> executing -> outcome
  useEffect(() => {
    setStage('selecting')
    const t1 = setTimeout(() => setStage('selected'), SELECT_MS)
    const t2 = setTimeout(() => setStage('executing'), SELECT_MS + 150)
    const t3 = setTimeout(() => setStage('outcome'), SELECT_MS + 150 + EXECUTE_MS)
    const done = setTimeout(() => onDone(data.outcome === 'success'), SELECT_MS + 150 + EXECUTE_MS + 500)
    return () => [t1, t2, t3, done].forEach(clearTimeout)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  const chosenLabel = RECOVERY_STRATEGIES.find((s) => s.id === data.chosen_strategy)?.label ?? data.chosen_strategy
  const success = data.outcome === 'success'
  const selected = stage !== 'selecting'
  const resolved = stage === 'outcome'

  return (
    <>
      <EventRow
        index={attemptNumber > 1 ? `retry ${attemptNumber}` : '04'}
        label="Strategy chosen"
        status={selected ? 'done' : 'active'}
        tone="adaptive"
      >
        {selected ? (
          <>
            <span className="font-medium text-lav-primary">{chosenLabel}</span>
            <div className="mt-2 max-w-sm">
              <StrategyBeliefBars
                before={data.strategy_scores_before}
                after={data.strategy_scores_after}
                chosen={data.chosen_strategy}
                revealAfter={resolved}
              />
            </div>
          </>
        ) : (
          <span className="text-lav-muted">sampling from belief...</span>
        )}
      </EventRow>

      {stage !== 'selecting' && (
        <EventRow
          index={attemptNumber > 1 ? '' : '05'}
          label="Outcome"
          status={resolved ? 'done' : 'active'}
          tone={!resolved ? 'live' : success ? 'success' : 'fail'}
        >
          {resolved ? (
            <span className={success ? 'font-medium text-lav-success' : 'font-medium text-lav-cost'}>
              {success ? 'Success' : 'Failed'}
            </span>
          ) : (
            <span className="text-lav-muted">executing {chosenLabel}...</span>
          )}
        </EventRow>
      )}
    </>
  )
}

/** The live event log for one injected failure: Task -> Detected ->
 * Classified -> (Strategy chosen -> Outcome) x attempts -> Recovered /
 * Escalated. Replaces the old horizontal numbered-circle stepper with a
 * single continuous vertical trace, the way a real system log or an
 * observability trace waterfall reads: append-only, oldest at top, every
 * past attempt stays visible instead of being replaced. */
export function EventStream({ result }) {
  const [classifierStage, setClassifierStage] = useState('processing')
  const [attempts, setAttempts] = useState([])
  const [retryLabel, setRetryLabel] = useState(null)
  const [finalOutcome, setFinalOutcome] = useState(null)

  useEffect(() => {
    if (!result || result.true_failure_type == null) return undefined
    setClassifierStage('processing')
    setAttempts([])
    setRetryLabel(null)
    setFinalOutcome(null)

    const t1 = setTimeout(() => setClassifierStage('revealed'), CLASSIFY_MS)
    const t2 = setTimeout(
      () => setAttempts([{ attemptNumber: 1, data: attempt1FromResult(result) }]),
      CLASSIFY_MS + 250,
    )
    return () => [t1, t2].forEach(clearTimeout)
  }, [result])

  if (!result || result.true_failure_type == null) return null

  const handleAttemptDone = (attemptNumber, success) => {
    if (success) {
      setFinalOutcome({ type: 'recovered', attempt: attemptNumber })
      return
    }
    if (attemptNumber >= MAX_ATTEMPTS) {
      setFinalOutcome({ type: 'escalated' })
      return
    }
    setRetryLabel(`Attempt ${attemptNumber} failed — resampling with updated belief`)
    setTimeout(async () => {
      try {
        const nextAttemptNumber = attemptNumber + 1
        const data = await retryRecovery(result.true_failure_type, result.predicted_failure_type, nextAttemptNumber)
        setAttempts((prev) => [...prev, { attemptNumber: nextAttemptNumber, data }])
      } catch {
        setFinalOutcome({ type: 'escalated' })
      } finally {
        setRetryLabel(null)
      }
    }, RETRY_LABEL_MS)
  }

  const trueType = FAILURE_TYPES.find((f) => f.id === result.true_failure_type)
  const predictedType = FAILURE_TYPES.find((f) => f.id === result.predicted_failure_type)

  return (
    <EventSpine>
      <EventRow index="01" label="Task" status="done" tone="neutral">
        {result.tool_called ? <span className="font-mono text-xs text-lav-muted">{result.tool_called}(...)</span> : 'agent action'}
      </EventRow>

      <EventRow index="02" label="Detected" status="done" tone="neutral">
        {trueType && (
          <span className="inline-flex items-center gap-1.5">
            <trueType.icon size={13} style={{ color: trueType.color }} />
            {trueType.label}
          </span>
        )}
      </EventRow>

      <EventRow index="03" label="Classified" status={classifierStage === 'revealed' ? 'done' : 'active'} tone="detect">
        {classifierStage === 'revealed' ? (
          <span className="inline-flex items-baseline gap-2">
            <span className="font-medium text-lav-text">{predictedType?.label ?? result.predicted_failure_type}</span>
            <span className="font-mono text-xs text-lav-amber">{Math.round((result.classifier_confidence ?? 0) * 100)}% confidence</span>
          </span>
        ) : (
          <span className="text-lav-muted">classifying...</span>
        )}
      </EventRow>

      {attempts.map((a) => (
        <AttemptEvents key={a.attemptNumber} attemptNumber={a.attemptNumber} data={a.data} onDone={(s) => handleAttemptDone(a.attemptNumber, s)} />
      ))}

      {retryLabel && (
        <EventRow label="Retrying" status="active" tone="detect">
          <span className="inline-flex items-center gap-1.5 text-lav-muted">
            <RotateCw size={12} className="animate-spin" />
            {retryLabel}
          </span>
        </EventRow>
      )}

      {finalOutcome?.type === 'recovered' && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-1 flex items-center gap-2 border-l-2 border-lav-success py-1.5 pl-3 font-sans text-sm font-medium text-lav-success"
        >
          <CheckCircle2 size={15} />
          Recovered on attempt {finalOutcome.attempt}
        </motion.div>
      )}

      {finalOutcome?.type === 'escalated' && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-1 flex items-center gap-2 border-l-2 border-lav-amber py-1.5 pl-3 font-sans text-sm font-medium text-lav-amber"
        >
          <UserCheck size={15} />
          Escalating to human, clarification requested
        </motion.div>
      )}
    </EventSpine>
  )
}
