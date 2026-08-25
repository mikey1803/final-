import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { CheckCircle2, UserCheck } from 'lucide-react'
import { TraceNode, TraceConnector } from './TraceNode'
import { PipelineConnector } from './PipelineLayer'
import { AttemptRound } from './AttemptRound'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'
import { FAILURE_TYPES } from '../../data/failureTypes'
import { retryRecovery } from '../../lib/demoApi'

const MAX_ATTEMPTS = 3
const L1_REVEAL_MS = 600
const L1_TO_ROUND1_MS = 300 // buffer between classifier reveal and attempt 1 starting
const RETRY_LABEL_MS = 700 // how long "Attempt N failed - retrying..." shows before the next round mounts

function attempt1FromResult(result) {
  return {
    strategy_scores_before: result.strategy_scores_before,
    strategy_scores_after: result.strategy_scores_after,
    chosen_strategy: result.chosen_strategy,
    outcome: result.outcome,
  }
}

export function PipelineView({ result }) {
  const [classifierStage, setClassifierStage] = useState('processing')
  const [attempts, setAttempts] = useState([]) // [{attemptNumber, data}]
  const [retryLabel, setRetryLabel] = useState(null) // "Attempt 1 failed - retrying with updated belief..."
  const [finalOutcome, setFinalOutcome] = useState(null) // {type: 'recovered'|'escalated', attempt}

  useEffect(() => {
    if (!result || result.true_failure_type == null) return undefined
    setClassifierStage('processing')
    setAttempts([])
    setRetryLabel(null)
    setFinalOutcome(null)

    const timers = []
    timers.push(setTimeout(() => setClassifierStage('revealed'), L1_REVEAL_MS))
    timers.push(
      setTimeout(() => {
        setAttempts([{ attemptNumber: 1, data: attempt1FromResult(result) }])
      }, L1_REVEAL_MS + L1_TO_ROUND1_MS),
    )
    return () => timers.forEach(clearTimeout)
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
    setRetryLabel(`Attempt ${attemptNumber} failed, retrying with updated belief...`)
    setTimeout(async () => {
      try {
        const nextAttemptNumber = attemptNumber + 1
        const data = await retryRecovery(result.true_failure_type, result.predicted_failure_type, nextAttemptNumber)
        setAttempts((prev) => [...prev, { attemptNumber: nextAttemptNumber, data }])
      } catch {
        // Backend hiccup mid-retry — treat like exhausting attempts rather than hanging forever.
        setFinalOutcome({ type: 'escalated' })
      } finally {
        setRetryLabel(null)
      }
    }, RETRY_LABEL_MS)
  }

  const chosenLabel = (id) => RECOVERY_STRATEGIES.find((s) => s.id === id)?.label ?? id
  const trueType = FAILURE_TYPES.find((f) => f.id === result.true_failure_type)
  const predictedType = FAILURE_TYPES.find((f) => f.id === result.predicted_failure_type)
  const [primaryAttempt, ...retryAttempts] = attempts

  return (
    <div className="flex w-full flex-col items-stretch gap-4">
      {/* Hero: the primary 5-node trace for attempt 1 — Task -> Detected ->
          Classified -> Strategy chosen -> Outcome. Horizontal on >=768px,
          stacks vertically below it (see TraceConnector). */}
      <div className="flex flex-col items-center gap-2 md:flex-row md:items-start md:justify-between md:gap-0">
        <TraceNode index={1} label="Task" status="done">
          {result.tool_called ? `${result.tool_called}(...)` : 'agent action'}
        </TraceNode>

        <TraceConnector />
        <TraceNode index={2} label="Detected" status="done">
          {trueType && (
            <>
              <trueType.icon size={12} className="inline" style={{ color: trueType.color }} /> {trueType.label}
            </>
          )}
        </TraceNode>

        <TraceConnector />
        <TraceNode index={3} label="Classified" status={classifierStage === 'revealed' ? 'done' : 'active'}>
          {classifierStage === 'revealed' ? (
            <>
              {predictedType?.label ?? result.predicted_failure_type}
              <span className="block text-lav-muted">{Math.round((result.classifier_confidence ?? 0) * 100)}%</span>
            </>
          ) : (
            'classifying...'
          )}
        </TraceNode>

        {primaryAttempt && (
          <AttemptRound
            attemptNumber={primaryAttempt.attemptNumber}
            data={primaryAttempt.data}
            isPrimary
            onDone={(success) => handleAttemptDone(primaryAttempt.attemptNumber, success)}
          />
        )}
      </div>

      {/* Retries (attempts 2-3): same trace nodes, smaller scale, stacked below. */}
      {retryAttempts.map((a) => (
        <AttemptRound
          key={a.attemptNumber}
          attemptNumber={a.attemptNumber}
          data={a.data}
          onDone={(success) => handleAttemptDone(a.attemptNumber, success)}
        />
      ))}

      {retryLabel && (
        <>
          <PipelineConnector />
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center font-mono text-xs text-lav-cost"
          >
            {retryLabel}
          </motion.p>
        </>
      )}

      {finalOutcome?.type === 'recovered' && (
        <>
          <PipelineConnector />
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 22 }}
            className="flex items-center justify-center gap-2 rounded-full bg-lav-success/15 px-5 py-2.5 text-center font-sans text-sm font-semibold text-lav-success"
          >
            <CheckCircle2 size={18} />
            Recovered on attempt {finalOutcome.attempt}
            {attempts.length > 0 && ` (${chosenLabel(attempts[attempts.length - 1].data.chosen_strategy)} worked)`}
          </motion.div>
        </>
      )}

      {finalOutcome?.type === 'escalated' && (
        <>
          <PipelineConnector />
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 22 }}
            className="flex items-center justify-center gap-2 rounded-full bg-lav-cost/15 px-5 py-2.5 text-center font-sans text-sm font-semibold text-lav-cost"
          >
            <UserCheck size={18} />
            Escalating to human, clarify requested
          </motion.div>
        </>
      )}
    </div>
  )
}
