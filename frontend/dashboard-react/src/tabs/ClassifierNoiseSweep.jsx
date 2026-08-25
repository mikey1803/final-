import {
  ComposedChart,
  Area,
  Line,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { Metric } from '../components/ui/Metric'
import { Figure } from '../components/ui/Figure'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { ChartTooltip } from '../components/ChartTooltip'
import { InteractiveLegend } from '../components/InteractiveLegend'
import { useCsv } from '../hooks/useCsv'
import { useSeriesToggle } from '../hooks/useSeriesToggle'
import { colorFor } from '../lib/colors'

const METHODS = ['Adaptive (Thompson)', 'Fixed Rule', 'Naive Retry']

// Task 19A — pre-registered, environment-level held-out kappa protocol
// result (docs/theory/kappa_preregistration.md, docs/theory/crossover_derivation.md
// §7.1). Fixed, already-computed values — not swept/fetched data.
const KAPPA_RESULT = {
  kappaA: 0.044381,
  kappaB: 0.042559,
  eA: 0.5970, // Branch A (transfer): median abs. rel. error on held-out C!=I TEST
  eB: 0.4053, // Branch B (in-condition ceiling): median abs. rel. error, same TEST cells
  R: 1.4729,
  threshold: 1.5,
  ciLow: 1.0,
  ciHigh: 9.55,
}

const ERROR_BY_BRANCH = [
  { branch: 'Branch A: transfer (kappa fit on C=I, applied to C≠I)', error: KAPPA_RESULT.eA },
  { branch: 'Branch B: in-condition (kappa fit directly on C≠I)', error: KAPPA_RESULT.eB },
]

// Branch A applied, unchanged, to the 8 pre-registered held-out TEST
// environments' C!=I cells (crossover_derivation.md §7.1).
const TEST_CELL_ROWS = [
  { env_id: 'gap0.20_idx00', ratio: 1.487, signed_rel_error: 0.487 },
  { env_id: 'gap0.20_idx01', ratio: 1.527, signed_rel_error: 0.527 },
  { env_id: 'gap0.20_idx06', ratio: 1.283, signed_rel_error: 0.283 },
  { env_id: 'gap0.20_idx08', ratio: 5.756, signed_rel_error: 4.756 },
  { env_id: 'gap0.20_idx11', ratio: 0.675, signed_rel_error: -0.325 },
  { env_id: 'gap0.20_idx16', ratio: 0.333, signed_rel_error: -0.667 },
  { env_id: 'gap0.20_idx17', ratio: 5.752, signed_rel_error: 4.752 },
  { env_id: 'gap0.20_idx24', ratio: 1.694, signed_rel_error: 0.694 },
]

function pct(v) {
  return `${(v * 100).toFixed(1)}%`
}

function pivotByAccuracy(rows) {
  const byAcc = new Map()
  for (const r of rows) {
    const key = r.effective_accuracy
    if (!byAcc.has(key)) byAcc.set(key, { effective_accuracy: key, beta: r.beta })
    const entry = byAcc.get(key)
    entry[r.method] = r.normalized_late_stage_score
    entry[`${r.method}__range`] = [r.bootstrap_ci95_lower_normalized, r.bootstrap_ci95_upper_normalized]
  }
  return [...byAcc.values()].sort((a, b) => a.effective_accuracy - b.effective_accuracy)
}

export function ClassifierNoiseSweep() {
  const { data, loading, error } = useCsv('context_sweep_results.csv')
  const series = useSeriesToggle()

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load sweep data.</p>

  const rows = pivotByAccuracy(data)

  return (
    <div>
      <PageHeader
        title="Classifier-Noise Sweep"
        subtitle="Synthetic classifier accuracy swept from 94% down to random guessing (beta: 1.0 -> 0.0). Fixed Rule and Thompson both condition on the noisy predicted failure type."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Late-stage % of Oracle vs. effective classification accuracy
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              Shaded bands = bootstrap 95% CI. Fixed Rule stays above Thompson at every accuracy
              level: the gap narrows but never closes.
            </p>
            <ResponsiveContainer width="100%" height={460}>
              <ComposedChart data={rows} margin={{ top: 8, right: 24, left: 0, bottom: 36 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="effective_accuracy"
                  type="number"
                  domain={[0.14, 0.94]}
                  tickFormatter={(v) => v.toFixed(1)}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  label={{
                    value: 'Effective classification accuracy',
                    position: 'bottom',
                    offset: 0,
                    style: { fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-muted)' },
                  }}
                />
                <YAxis
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  label={{
                    value: 'Normalized score',
                    angle: -90,
                    position: 'insideLeft',
                    style: { fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-muted)' },
                  }}
                />
                <Tooltip
                  content={
                    <ChartTooltip
                      formatter={(v) => v.toFixed(3)}
                      labelFormatter={(v) => `accuracy = ${Number(v).toFixed(3)}`}
                    />
                  }
                  cursor={{ stroke: 'var(--color-lav-primary)', strokeWidth: 1, strokeDasharray: '4 4' }}
                />
                <Legend
                  verticalAlign="top"
                  height={40}
                  content={
                    <InteractiveLegend hidden={series.hidden} onToggle={series.toggle} onHover={series.setHovered} />
                  }
                />
                <ReferenceLine y={0} stroke="var(--color-lav-muted)" strokeDasharray="3 3" />
                <ReferenceLine y={1} stroke="var(--color-lav-muted)" strokeDasharray="3 3" />

                {METHODS.map((m) => (
                  <Area
                    key={`${m}-band`}
                    dataKey={`${m}__range`}
                    stroke="none"
                    fill={colorFor(m)}
                    fillOpacity={series.hoveredKey && series.hoveredKey !== m ? 0.04 : 0.12}
                    hide={series.isHidden(m)}
                    animationDuration={500}
                    animationEasing="ease-out"
                    legendType="none"
                    connectNulls
                  />
                ))}
                {METHODS.map((m) => (
                  <Line
                    key={m}
                    dataKey={m}
                    name={m}
                    stroke={colorFor(m)}
                    strokeWidth={2.4}
                    strokeOpacity={series.hoveredKey && series.hoveredKey !== m ? 0.3 : 1}
                    dot={{ r: 3, fill: colorFor(m), strokeWidth: 0 }}
                    hide={series.isHidden(m)}
                    animationDuration={500}
                    animationEasing="ease-out"
                    connectNulls
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <details className="group">
            <summary className="cursor-pointer select-none font-sans text-sm text-lav-muted transition-colors hover:text-lav-primary">
              Show original static figure (Matplotlib reference)
            </summary>
            <div className="mt-3">
              <Figure
                src={`${import.meta.env.BASE_URL}figures/context_sweep.png`}
                alt="Classifier-noise sweep, original Matplotlib figure"
                caption="results/phase4b/context_sweep.png"
              />
            </div>
          </details>
        </StaggerItem>

        <StaggerItem>
          <div className="mt-4 mb-1 flex items-center gap-3">
            <div className="h-px flex-1 bg-lav-border" />
            <span className="font-sans text-xs font-semibold uppercase tracking-wide text-lav-muted">
              Task 19A: Classifier-Noise-Aware Crossover Theory
            </span>
            <div className="h-px flex-1 bg-lav-border" />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h3 className="font-sans text-base font-semibold text-lav-text">
                Does a classifier-noise-aware crossover predictor generalize across environments?
              </h3>
              <span className="rounded-full border border-lav-cost/40 bg-lav-cost/10 px-3 py-1 font-sans text-xs font-semibold tracking-wide text-lav-cost">
                PARTIALLY-SUPPORTED
              </span>
            </div>
            <p className="font-sans text-sm leading-relaxed text-lav-text">
              In plain terms: the earlier prediction formula assumed the agent could see the true
              failure type. It cannot, it only sees a classifier's guess, so this section checks
              whether the formula still holds once that's accounted for.
            </p>
            <p className="mt-3 font-sans text-sm leading-relaxed text-lav-text">
              The sweep above shows adaptive vs. static performance as classifier accuracy degrades.
              Task 19A goes further: it re-derives Task 10's analytic crossover predictor (the horizon{' '}
              <span className="font-mono text-lav-primary">T*</span> at which Thompson Sampling
              overtakes a static table) to account for the fact that every policy in this codebase only
              ever observes a classifier's <em>predicted</em> failure type, never the true one, via an
              effective mixture matrix{' '}
              <span className="font-mono text-lav-primary">P_eff</span> (Bayes-flipping the confusion
              matrix against the type prior). The predictor still carries one free constant,{' '}
              <span className="font-mono text-lav-primary">kappa</span>, that has to be fit empirically.
              The question below is whether a kappa fit under one classifier condition transfers to
              another, tested with a pre-registered, environment-level held-out protocol (not an
              in-sample fit).
            </p>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Metric label="kappa (transfer, Branch A)" value={KAPPA_RESULT.kappaA.toFixed(4)} accent="primary" />
            <Metric label="kappa (in-condition, Branch B)" value={KAPPA_RESULT.kappaB.toFixed(4)} accent="primary" />
            <Metric label="R = E_A / E_B" value={KAPPA_RESULT.R.toFixed(4)} accent="cost" help="Decision statistic vs. the pre-committed threshold of 1.5" />
            <Metric label="Held-out cells (train / test)" value="18 / 8" accent="success" help="Environment-level split, KAPPA_SPLIT_SEED=19001" />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Held-out error: transfer (Branch A) vs. best achievable same-condition fit (Branch B)
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              Both bars are evaluated on the SAME 8 held-out environments. Branch A applies a kappa
              fit on the 18 TRAIN environments&apos; perfect-classifier (C=I) cells, unchanged, to the
              TEST environments&apos; real-classifier (C≠I) cells. Branch B fits directly on C≠I. R =
              A&apos;s error / B&apos;s error = {KAPPA_RESULT.R.toFixed(4)}, under the pre-committed
              threshold of {KAPPA_RESULT.threshold} by 1.8%.
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ERROR_BY_BRANCH} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
                <CartesianGrid stroke="var(--color-lav-border)" horizontal={false} />
                <XAxis
                  type="number"
                  tickFormatter={pct}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  type="category"
                  dataKey="branch"
                  width={260}
                  tick={{ fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-text)' }}
                  stroke="var(--color-lav-border)"
                />
                <Tooltip content={<ChartTooltip formatter={pct} />} cursor={{ fill: 'rgba(167,139,250,0.08)' }} />
                <Bar dataKey="error" name="Median abs. rel. error" radius={[0, 6, 6, 0]} animationDuration={500} animationEasing="ease-out" fill={colorFor('Adaptive (Thompson)')} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              R&apos;s pre-registered bootstrap 95% CI vs. the decision threshold
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              2,000 environment-level resamples of the 8 held-out TEST environments, reusing the
              already-fitted kappas. The point estimate clears the threshold; the interval does not
              confirm that clearance is robust: it reaches well past the Case-2 boundary.
            </p>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart
                data={[{ name: 'R, bootstrap 95% CI', base: KAPPA_RESULT.ciLow, range: KAPPA_RESULT.ciHigh - KAPPA_RESULT.ciLow }]}
                layout="vertical"
                margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
              >
                <CartesianGrid stroke="var(--color-lav-border)" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 10]}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={150}
                  tick={{ fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-text)' }}
                  stroke="var(--color-lav-border)"
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => v.toFixed(2)} />}
                  cursor={{ fill: 'rgba(167,139,250,0.08)' }}
                />
                <ReferenceLine
                  x={KAPPA_RESULT.threshold}
                  stroke="var(--color-lav-cost)"
                  strokeDasharray="4 4"
                  label={{ value: `threshold = ${KAPPA_RESULT.threshold}`, fill: 'var(--color-lav-cost)', fontSize: 11, position: 'top' }}
                />
                <ReferenceLine
                  x={KAPPA_RESULT.R}
                  stroke="var(--color-lav-success)"
                  strokeWidth={2}
                  label={{ value: `R = ${KAPPA_RESULT.R.toFixed(2)}`, fill: 'var(--color-lav-success)', fontSize: 11, position: 'insideTopRight' }}
                />
                <Bar dataKey="base" stackId="ci" fill="transparent" animationDuration={500} animationEasing="ease-out" />
                <Bar dataKey="range" stackId="ci" fill="var(--color-lav-primary)" fillOpacity={0.35} radius={[6, 6, 6, 6]} animationDuration={500} animationEasing="ease-out" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Per-environment residuals, Branch A on the 8 held-out TEST environments
            </h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              Highly heterogeneous: two environments (idx08, idx17) are large over-predictions under
              both kappas; idx16 is a consistent under-prediction. No single scalar kappa corrects this
              spread, and no attempt was made to (pre-registration&apos;s no-rescue commitment).
            </p>
            <DataTable
              columns={[
                { key: 'env_id', label: 'Environment' },
                { key: 'ratio', label: 'predicted / observed', align: 'right', render: (r) => r.ratio.toFixed(3) },
                {
                  key: 'signed_rel_error',
                  label: 'Signed rel. error',
                  align: 'right',
                  render: (r) => (
                    <span className={r.signed_rel_error >= 0 ? 'text-lav-cost' : 'text-lav-primary'}>
                      {r.signed_rel_error >= 0 ? '+' : ''}
                      {pct(r.signed_rel_error)}
                    </span>
                  ),
                },
              ]}
              rows={TEST_CELL_ROWS}
              sortable
            />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <p className="font-sans text-sm leading-relaxed text-lav-text">
              <strong>Reported outcome (pre-committed wording):</strong> &ldquo;P_eff captures the
              classifier-noise effect; the residual prediction error is inherited Task 10
              asymptotic-bound bias, not error introduced by this extension.&rdquo; This is a claim{' '}
              <em>relative to</em> Task 10&apos;s existing bound. Absolute error stays large (
              {pct(KAPPA_RESULT.eA)} / {pct(KAPPA_RESULT.eB)} median on the two branches), so Task
              10&apos;s bound is not newly accurate in absolute terms. The fitted kappa (
              {KAPPA_RESULT.kappaA.toFixed(4)}) falls within one order of magnitude of Task 10&apos;s
              own calibrated constant, so this is described as a{' '}
              <strong>calibrated predictor</strong>, not a validated bound. Classified{' '}
              <strong>PARTIALLY-SUPPORTED</strong> rather than VALIDATED: the pre-registered decision
              rule is satisfied by the point estimate, but the same protocol&apos;s own bootstrap does
              not confirm that satisfaction is robust to which environments are sampled. Full protocol:{' '}
              <span className="font-mono text-lav-muted">docs/theory/kappa_preregistration.md</span>;
              full result: <span className="font-mono text-lav-muted">docs/theory/crossover_derivation.md</span> §7.1.
            </p>
          </Card>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
