import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { CheckCircle2 } from 'lucide-react'
import { FAILURE_TYPES } from '../data/failureTypes'
import { RECOVERY_STRATEGIES, BELIEF_EXAMPLES } from '../data/recoveryStrategies'
import { Card } from '../components/ui/Card'

const STAGE_LABELS = ['Failure occurs', 'Consider strategies', 'Pick one', 'Outcome', 'Update belief']
const STAGE_DURATIONS = [1000, 1000, 1100, 1000, 2600]

const strategyContainer = { hidden: {}, show: { transition: { staggerChildren: 0.06 } } }
const strategyItem = { hidden: { opacity: 0, scale: 0.85 }, show: { opacity: 1, scale: 1 } }

export function LearningLoopDiagram() {
  const [exampleIndex, setExampleIndex] = useState(0)
  const [stage, setStage] = useState(0)
  const timerRef = useRef(null)

  useEffect(() => {
    let curStage = 0
    let curExample = 0

    const schedule = () => {
      timerRef.current = setTimeout(() => {
        if (curStage < STAGE_LABELS.length - 1) {
          curStage += 1
        } else {
          curExample = (curExample + 1) % BELIEF_EXAMPLES.length
          curStage = 0
          setExampleIndex(curExample)
        }
        setStage(curStage)
        schedule()
      }, STAGE_DURATIONS[curStage])
    }
    schedule()
    return () => clearTimeout(timerRef.current)
  }, [])

  const example = BELIEF_EXAMPLES[exampleIndex]
  const failureType = FAILURE_TYPES.find((f) => f.id === example.failureTypeId)

  return (
    <Card hover={false} className="overflow-hidden">
      <div className="mb-8 flex flex-wrap items-center justify-center gap-x-1.5 gap-y-2">
        {STAGE_LABELS.map((label, i) => (
          <div key={label} className="flex items-center gap-1.5">
            <span
              className={
                'rounded-full px-3 py-1 font-sans text-xs transition-colors duration-300 ' +
                (i > stage ? 'bg-lav-bg text-lav-muted' : '')
              }
              style={
                i === stage
                  ? { backgroundColor: 'var(--color-lav-teal)', color: 'var(--color-lav-bg)' }
                  : i < stage
                    ? { backgroundColor: 'rgba(94,234,212,0.15)', color: 'var(--color-lav-teal)' }
                    : undefined
              }
            >
              {i + 1}. {label}
            </span>
            {i < STAGE_LABELS.length - 1 && <span className="text-lav-border">&rarr;</span>}
          </div>
        ))}
      </div>

      <div className="flex min-h-[240px] flex-col items-center justify-center gap-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={exampleIndex}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="flex items-center gap-2 rounded-full px-4 py-2"
            style={{ backgroundColor: `${failureType.color}1A`, color: failureType.color }}
          >
            <failureType.icon size={16} />
            <span className="font-sans text-sm font-medium">{failureType.label}</span>
          </motion.div>
        </AnimatePresence>

        {stage >= 1 && (
          <motion.div
            variants={strategyContainer}
            initial="hidden"
            animate="show"
            className="flex flex-wrap justify-center gap-2"
          >
            {RECOVERY_STRATEGIES.map((s) => {
              const isChosen = s.id === example.winner
              const highlighted = stage >= 2 && isChosen
              return (
                <motion.span
                  key={s.id}
                  variants={strategyItem}
                  animate={highlighted ? { scale: 1.08 } : { scale: 1 }}
                  className={
                    'rounded-full border px-3 py-1.5 font-mono text-xs transition-colors duration-300 ' +
                    (highlighted
                      ? 'shadow-[0_2px_8px_rgba(94,234,212,0.22)]'
                      : 'border-lav-border bg-lav-surface text-lav-muted')
                  }
                  style={
                    highlighted
                      ? { borderColor: 'var(--color-lav-teal)', backgroundColor: 'var(--color-lav-teal)', color: 'var(--color-lav-bg)' }
                      : undefined
                  }
                >
                  {s.label}
                </motion.span>
              )
            })}
          </motion.div>
        )}

        {stage >= 3 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-1.5 font-sans text-sm font-medium text-lav-success"
          >
            <CheckCircle2 size={16} />
            It worked. Outcome recorded.
          </motion.div>
        )}

        {stage >= 4 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full max-w-md">
            <p className="mb-2 text-center font-sans text-xs uppercase tracking-wide text-lav-muted">
              Updated belief for “{failureType.label}”
            </p>
            <div className="flex flex-col gap-1.5">
              {RECOVERY_STRATEGIES.map((s) => {
                const value = example.beliefs[s.id]
                const isWinner = s.id === example.winner
                return (
                  <div key={s.id} className="flex items-center gap-2">
                    <span className="w-20 shrink-0 text-right font-mono text-[11px] text-lav-muted">
                      {s.label}
                    </span>
                    <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-lav-bg">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${value * 100}%` }}
                        transition={{ duration: 0.6, ease: 'easeOut' }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: isWinner ? 'var(--color-lav-teal)' : 'var(--color-lav-surface-raised)' }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </div>
    </Card>
  )
}
