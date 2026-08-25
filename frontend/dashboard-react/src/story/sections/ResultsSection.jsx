import { motion } from 'motion/react'
import { Lock, TrendingUp, LayoutGrid } from 'lucide-react'
import { Card } from '../../components/ui/Card'
import { Figure } from '../../components/ui/Figure'
import { DataTable } from '../../components/ui/DataTable'
import { useCsv } from '../../hooks/useCsv'
import { RevealText } from '../RevealText'

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

const columns = [
  { key: 'method', label: 'Method' },
  {
    key: 'rate',
    label: 'Late-stage success rate',
    align: 'right',
    render: (r) => `${(r.mean_late_stage_success_rate * 100).toFixed(1)}%`,
  },
  {
    key: 'norm',
    label: 'Normalized score',
    align: 'right',
    render: (r) => (
      <span className={r.normalized_late_stage_score >= 0 ? 'text-lav-success' : 'text-lav-cost'}>
        {r.normalized_late_stage_score >= 0 ? '+' : ''}
        {r.normalized_late_stage_score.toFixed(3)}
      </span>
    ),
  },
]

export function ResultsSection({ onOpenDashboard }) {
  const { data } = useCsv('comparison_summary.csv')
  const rows = data ? [...data].sort((a, b) => b.normalized_late_stage_score - a.normalized_late_stage_score) : []

  return (
    <section id="results" className="scroll-mt-24 py-20">
      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.5 }}
        variants={fadeUp}
        className="mx-auto max-w-2xl text-center"
      >
        <RevealText as="h2" className="font-heading text-4xl font-semibold text-lav-text sm:text-5xl">
          Here’s what we found
        </RevealText>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.4 }}
        variants={fadeUp}
        className="mx-auto mt-8 max-w-2xl"
      >
        <Card hover={false} className="divide-y divide-lav-border">
          <div className="flex items-start gap-3 py-3 first:pt-0">
            <Lock size={16} className="mt-0.5 shrink-0 text-lav-cost" />
            <p className="font-sans text-sm text-lav-text">
              Static rules win under perfect conditions, but can’t adapt when things change.
            </p>
          </div>
          <div className="flex items-start gap-3 py-3 last:pb-0">
            <TrendingUp size={16} className="mt-0.5 shrink-0 text-lav-success" />
            <p className="font-sans text-sm text-lav-text">
              Online learning is slower at first, but recovers when the world shifts.
            </p>
          </div>
        </Card>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.25 }}
        variants={fadeUp}
        className="mx-auto mt-14 max-w-3xl"
      >
        <Figure
          src={`${import.meta.env.BASE_URL}figures/context_sweep.png`}
          alt="Classifier-noise sweep: normalized score vs. classifier accuracy for Thompson, Fixed Rule, and Naive Retry"
          caption="results/phase4b/context_sweep.png"
        />
        <p className="mt-3 font-sans text-sm leading-relaxed text-lav-muted">
          <strong className="text-lav-text">What this tests:</strong> what happens when the
          failure classifier feeding the selector isn’t perfect. Accuracy is swept from 94% down
          to random guessing. <strong className="text-lav-text">What we found:</strong> the Fixed
          Rule stays ahead of Thompson Sampling at every noise level; the gap narrows but never
          closes. <strong className="text-lav-text">Why it matters:</strong> a real classifier is
          never perfect, so this shows the static rule's advantage isn't just a lab artifact; it
          holds up under realistic noise.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.25 }}
        variants={fadeUp}
        className="mx-auto mt-14 max-w-3xl"
      >
        <Figure
          src={`${import.meta.env.BASE_URL}figures/nonstationarity_recovery_curve.png`}
          alt="Rolling success rate before and after a simulated mid-run disruption at episode 250"
          caption="results/phase4b/nonstationarity_recovery_curve.png"
        />
        <p className="mt-3 font-sans text-sm leading-relaxed text-lav-muted">
          <strong className="text-lav-text">What this tests:</strong> a simulated tool/API
          degradation injected mid-run, at episode 250 of 500.{' '}
          <strong className="text-lav-text">What we found:</strong> the Fixed Rule starts highest
          (its blind design happens to be right for 6 of 7 failure types) but drops hardest after
          the shift and never recovers; Thompson Sampling dips less and climbs back up.{' '}
          <strong className="text-lav-text">Why it matters:</strong> this is the core trade-off of
          the whole project: a rule that can't reconsider is only as good as the assumptions it
          was built on.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.2 }}
        variants={fadeUp}
        className="mx-auto mt-14 max-w-3xl"
      >
        <h3 className="mb-1 font-sans text-xl font-semibold text-lav-text">
          Final comparison: all 11 methods
        </h3>
        <p className="mb-4 font-sans text-sm text-lav-muted">
          Normalized score: 0 = UniformRandom floor, 1 = Oracle ceiling. 500 episodes × 50 trials,
          stationary environment.
        </p>
        {rows.length > 0 && <DataTable columns={columns} rows={rows} animateRows sortable />}
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.6 }}
        variants={fadeUp}
        className="mx-auto mt-16 flex max-w-2xl flex-col items-center gap-3 text-center"
      >
        <p className="font-sans text-sm text-lav-muted">
          That's the story. Everything above, plus the matrix-family robustness checks and
          full statistical breakdowns, is in the full data dashboard.
        </p>
        <motion.button
          onClick={onOpenDashboard}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.97 }}
          transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
          className="accent-gradient flex items-center gap-2 rounded-full px-5 py-2.5 font-sans text-sm font-medium text-lav-bg shadow-[0_6px_20px_rgba(0,0,0,0.35)]"
        >
          <LayoutGrid size={16} />
          Explore the full data dashboard
        </motion.button>
      </motion.div>
    </section>
  )
}
