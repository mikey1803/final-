import { motion } from 'motion/react'
import { CheckCircle2, XCircle, Lock, TrendingUp } from 'lucide-react'
import { Card } from '../components/ui/Card'

const RULE_SEQUENCE = ['Retry', 'Retry', 'Retry', 'Retry', 'Retry']
const LEARN_SEQUENCE = [
  { label: 'Try A', ok: false },
  { label: 'Try B', ok: false },
  { label: 'Try C', ok: true },
  { label: 'Use C', ok: true },
  { label: 'Use C', ok: true },
]

const chipContainer = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12, delayChildren: 0.2 } },
}
const chipItem = {
  hidden: { opacity: 0, scale: 0.8 },
  show: { opacity: 1, scale: 1, transition: { duration: 0.25, ease: 'easeOut' } },
}

export function StuckVsAdaptive() {
  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
      <Card hover={false} className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-lav-cost/15 text-lav-cost">
            <Lock size={16} />
          </span>
          <h3 className="font-sans text-base font-semibold text-lav-text">Static Rule</h3>
        </div>
        <p className="mt-2 font-sans text-sm leading-relaxed text-lav-muted">
          A hand-designed rule always reacts to a given failure the same way. If that reaction is
          wrong for this failure type, it fails <em>forever</em>. It never reconsiders.
        </p>
        <motion.div
          variants={chipContainer}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.6 }}
          className="mt-4 flex flex-wrap items-center gap-2"
        >
          {RULE_SEQUENCE.map((label, i) => (
            <motion.span
              key={i}
              variants={chipItem}
              className="flex items-center gap-1.5 rounded-full border border-lav-cost/30 bg-lav-cost/10 px-3 py-1.5 font-mono text-xs text-lav-cost"
            >
              <XCircle size={13} />
              {label}
            </motion.span>
          ))}
        </motion.div>
        <p className="mt-3 font-sans text-xs italic text-lav-muted">
          Same action, same outcome, every single time.
        </p>
      </Card>

      <Card hover={false} className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-lav-success/15 text-lav-success">
            <TrendingUp size={16} />
          </span>
          <h3 className="font-sans text-base font-semibold text-lav-text">Online Learning</h3>
        </div>
        <p className="mt-2 font-sans text-sm leading-relaxed text-lav-muted">
          An adaptive selector tries different strategies, watches what actually works, and shifts
          its behavior as evidence comes in.
        </p>
        <motion.div
          variants={chipContainer}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.6 }}
          className="mt-4 flex flex-wrap items-center gap-2"
        >
          {LEARN_SEQUENCE.map((step, i) => (
            <motion.span
              key={i}
              variants={chipItem}
              className={
                'flex items-center gap-1.5 rounded-full border px-3 py-1.5 font-mono text-xs ' +
                (step.ok
                  ? 'border-lav-success/30 bg-lav-success/10 text-lav-success'
                  : 'border-lav-border bg-lav-bg text-lav-muted')
              }
            >
              {step.ok ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
              {step.label}
            </motion.span>
          ))}
        </motion.div>
        <p className="mt-3 font-sans text-xs italic text-lav-muted">
          Learns which strategy works, then keeps using it.
        </p>
      </Card>
    </div>
  )
}
