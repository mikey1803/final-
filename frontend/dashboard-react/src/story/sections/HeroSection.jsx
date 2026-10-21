import { useEffect, useState } from 'react'
import { motion, AnimatePresence, useReducedMotion } from 'motion/react'
import { ArrowRight } from 'lucide-react'
import { Card } from '../../components/ui/Card'
import { RevealText } from '../RevealText'
import { FAILURE_TYPES } from '../../data/failureTypes'
import { RECOVERY_STRATEGIES, BELIEF_EXAMPLES } from '../../data/recoveryStrategies'

const CYCLE_MS = 3600

/** Small live-feeling preview built from the same real converged-belief
 * numbers the Solution section explains in full (src/data/recoveryStrategies.js,
 * lifted from the actual SUCCESS_MATRIX) — a teaser of the real mechanic, not
 * a fabricated product screenshot. */
function BeliefPreviewCard() {
  const reduce = useReducedMotion()
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (reduce) return undefined
    const id = setInterval(() => setIndex((i) => (i + 1) % BELIEF_EXAMPLES.length), CYCLE_MS)
    return () => clearInterval(id)
  }, [reduce])

  const example = BELIEF_EXAMPLES[index]
  const failureType = FAILURE_TYPES.find((f) => f.id === example.failureTypeId)

  return (
    <Card hover={false} className="w-full max-w-sm p-6">
      <p className="font-sans text-xs font-medium uppercase tracking-wide text-lav-muted">
        Learned belief, for
      </p>
      <AnimatePresence mode="wait">
        <motion.div
          key={example.failureTypeId}
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 6 }}
          transition={{ duration: 0.3 }}
          className="mt-1 flex items-center gap-2"
          style={{ color: failureType.color }}
        >
          <failureType.icon size={16} />
          <span className="font-sans text-sm font-semibold">{failureType.label}</span>
        </motion.div>
      </AnimatePresence>

      <div className="mt-5 flex flex-col gap-2.5">
        {RECOVERY_STRATEGIES.map((s) => {
          const value = example.beliefs[s.id]
          const isWinner = s.id === example.winner
          return (
            <div key={s.id} className="flex items-center gap-2.5">
              <span className="w-[4.5rem] shrink-0 text-right font-mono text-[11px] text-lav-muted">
                {s.label}
              </span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-lav-bg">
                <motion.div
                  animate={{ width: `${value * 100}%` }}
                  transition={{ duration: 0.7, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{
                    backgroundColor: isWinner ? 'var(--color-lav-primary)' : 'var(--color-lav-surface-raised)',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <p className="mt-4 font-sans text-[11px] leading-relaxed text-lav-muted">
        Real numbers from the hidden success matrix, learned purely from trial and error.
      </p>
    </Card>
  )
}

export function HeroSection() {
  return (
    <section
      id="hero"
      className="scroll-mt-24 py-16 lg:flex lg:min-h-[calc(100dvh-4.5rem)] lg:items-center lg:py-0"
    >
      <div className="grid w-full grid-cols-1 items-center gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:gap-8">
        <div>
          <p className="font-sans text-xs font-semibold uppercase tracking-widest text-lav-primary">
            Final-year AI &amp; Data Science research
          </p>
          <RevealText
            as="h1"
            className="mt-3 font-heading text-5xl font-semibold leading-[1.05] text-lav-text sm:text-6xl"
          >
            Static rules win. Until the world changes.
          </RevealText>
          <p className="mt-5 max-w-prose font-sans text-base leading-relaxed text-lav-muted">
            An adaptive recovery system for LLM agent tool failures, tested across nine
            experimental phases and 500 episodes per trial.
          </p>
          <div className="mt-8">
            <motion.button
              onClick={() => document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' })}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
              className="flex items-center gap-2 rounded bg-lav-primary px-5 py-2.5 font-sans text-sm font-medium text-white"
            >
              Watch it learn
              <ArrowRight size={15} />
            </motion.button>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut', delay: 0.15 }}
          className="flex justify-center lg:justify-end"
        >
          <BeliefPreviewCard />
        </motion.div>
      </div>
    </section>
  )
}
