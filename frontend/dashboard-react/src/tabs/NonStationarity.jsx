import { useState } from 'react'
import {
  BarChart,
  Bar,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
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
import { useJson } from '../hooks/useJson'
import { useSeriesToggle } from '../hooks/useSeriesToggle'
import { colorFor } from '../lib/colors'

/** Rolling-window recovery curve, shared shape by both non-stationarity
 * charts below (all-episodes vs. shifted-type-only) — same band+line
 * ComposedChart pattern as ClassifierNoiseSweep/MatrixFamilySensitivity, so
 * it gets the same interactive legend (click to hide a method, hover to
 * dim the rest) instead of a static Matplotlib PNG. */
function RecoveryCurveChart({ data, xKey, xLabel, shiftX, shiftLabel, tooltipLabel, height = 340 }) {
  const series = useSeriesToggle()
  if (!data) return <div className="flex h-[340px] items-center justify-center text-sm text-lav-muted">Loading…</div>

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data.rows} margin={{ top: 8, right: 16, left: 0, bottom: 30 }}>
        <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
        <XAxis
          dataKey={xKey}
          type="number"
          domain={['dataMin', 'dataMax']}
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
          stroke="var(--color-lav-border)"
          label={{
            value: xLabel,
            position: 'bottom',
            offset: 4,
            style: { fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-muted)' },
          }}
        />
        <YAxis
          domain={[0, 1]}
          allowDataOverflow
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
          stroke="var(--color-lav-border)"
        />
        <Tooltip
          content={
            <ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} labelFormatter={(v) => `${tooltipLabel} ${v}`} />
          }
          cursor={{ stroke: 'var(--color-lav-primary)', strokeWidth: 1, strokeDasharray: '4 4' }}
        />
        <Legend
          verticalAlign="top"
          height={40}
          content={<InteractiveLegend hidden={series.hidden} onToggle={series.toggle} onHover={series.setHovered} />}
        />
        {shiftX != null && (
          <ReferenceLine
            x={shiftX}
            stroke="var(--color-lav-cost)"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{ value: shiftLabel, angle: -90, position: 'insideTopRight', fontSize: 10, fill: 'var(--color-lav-cost)' }}
          />
        )}
        {data.methods.map((m) => (
          <Area
            key={`${m}-band`}
            dataKey={`${m}__range`}
            stroke="none"
            fill={colorFor(data.methodLabels[m])}
            fillOpacity={series.hoveredKey && series.hoveredKey !== m ? 0.04 : 0.14}
            hide={series.isHidden(m)}
            animationDuration={500}
            animationEasing="ease-out"
            legendType="none"
            connectNulls
          />
        ))}
        {data.methods.map((m) => (
          <Line
            key={m}
            type="monotone"
            dataKey={m}
            name={data.methodLabels[m]}
            stroke={colorFor(data.methodLabels[m])}
            strokeWidth={2.2}
            strokeOpacity={series.hoveredKey && series.hoveredKey !== m ? 0.3 : 1}
            dot={false}
            hide={series.isHidden(m)}
            animationDuration={500}
            animationEasing="ease-out"
            connectNulls
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// Task 19B — drift-crossover negative result (docs/theory/drift_crossover_negative.md).
// Fixed, already-computed values from the pre-registered attempts, not swept data.
const ATTEMPTS = [
  { attempt: '1: (s, g) survival & gap-change', stat: 'R² = 0.012', verdict: 'Insufficient' },
  { attempt: '2: (d, τ) magnitude & recovery-ratio', stat: 'R² = 0.579-0.735', verdict: 'Retracted: conceptually circular' },
  { attempt: '3: (D, R) regret & reversion', stat: '79.7% = majority baseline', verdict: 'No quantitative rule survives leave-one-out' },
]

// Per-drift-type reversion fraction (R) and outcome, as reported in
// drift_crossover_negative.md §5–§6. partial_reorder/permutation R values
// are the paper's own per-type aggregates (P1); monotone_decay/gradual_rotation
// are the R≈0 population the confirmed finding is built on.
const REVERSION_BY_TYPE = [
  { drift_type: 'partial_reorder', R: 0.9412, negative: 10, total: 16 },
  { drift_type: 'permutation', R: 0.7217, negative: 3, total: 16 },
  { drift_type: 'gradual_rotation', R: 0.02, negative: 0, total: 16 },
  { drift_type: 'monotone_decay', R: 0.0, negative: 0, total: 16 },
]

export function NonStationarity() {
  const { data, loading, error } = useCsv('nonstationarity_adaptation.csv')
  const { data: recoveryCurve } = useJson('nonstationarity_recovery_curve.json')
  const { data: shiftedTypeOnly } = useJson('nonstationarity_shifted_type_only.json')
  const shiftSeries = useSeriesToggle()
  const [hoveredDelta, setHoveredDelta] = useState(null)
  const [hoveredReversion, setHoveredReversion] = useState(null)

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load adaptation summary.</p>

  const rows = data.map((r) => ({
    ...r,
    errY: [r.recovery_delta - r.recovery_delta_ci_lo, r.recovery_delta_ci_hi - r.recovery_delta],
  }))

  return (
    <div>
      <PageHeader
        title="Non-Stationarity"
        subtitle="Reward distribution shifted mid-run (simulated tool/API degradation at episode 250 of 500) for 2 of 7 failure types. Metrics below are on shifted-type episodes only."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card hover={false}>
              <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
                Stock TS vs Discounted TS vs Fixed Rule (blind)
              </h3>
              <RecoveryCurveChart
                data={recoveryCurve}
                xKey="episode"
                xLabel="Episode"
                shiftX={recoveryCurve?.shiftEpisode}
                shiftLabel="Shift"
                tooltipLabel="episode"
              />
              <p className="mt-2 text-center font-sans text-xs text-lav-muted">
                All-types rolling success rate (dilutes the shift's effect since only 2/7 types are affected).
                Shaded band = ±1 std across 50 trials. Hover a line to isolate it, click a legend entry to hide it.
              </p>
            </Card>
            <Card hover={false}>
              <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
                Shifted-type-only recovery curve
              </h3>
              <RecoveryCurveChart
                data={shiftedTypeOnly}
                xKey="occurrence"
                xLabel="Occurrence index (exposures to a shifted failure type)"
                shiftX={shiftedTypeOnly?.meanShiftOccurrence}
                shiftLabel="Shift"
                tooltipLabel="occurrence"
              />
              <p className="mt-2 text-center font-sans text-xs text-lav-muted">
                Shifted-types-only rolling success rate (isolates the recovery signal). x-axis = the Nth time a
                trial hits a shifted failure type, not the episode number.
              </p>
            </Card>
          </div>

          <details className="group mt-3">
            <summary className="cursor-pointer select-none font-sans text-sm text-lav-muted transition-colors hover:text-lav-primary">
              Show original static figures (Matplotlib reference)
            </summary>
            <div className="mt-3 grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Figure
                src={`${import.meta.env.BASE_URL}figures/nonstationarity_recovery_curve.png`}
                alt="Recovery curve across all failure types, with shift marker"
                caption="results/phase4b/nonstationarity_recovery_curve.png"
              />
              <Figure
                src={`${import.meta.env.BASE_URL}figures/nonstationarity_shifted_type_only.png`}
                alt="Recovery curve restricted to shifted failure types"
                caption="results/phase4b/nonstationarity_shifted_type_only.png"
              />
            </div>
          </details>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Recovery delta (late-post-shift minus early-post-shift)
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              The adaptation signal: a static rule cannot re-learn after the shift, so its recovery
              delta should sit at ~0. Error bars = bootstrap 95% CI.
            </p>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={rows} margin={{ top: 8, right: 24, left: 0, bottom: 24 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="method"
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => v.toFixed(3)} />}
                  cursor={{ fill: 'rgba(167,139,250,0.08)' }}
                />
                <ReferenceLine y={0} stroke="var(--color-lav-text)" />
                <Bar
                  dataKey="recovery_delta"
                  name="Recovery delta"
                  radius={[6, 6, 0, 0]}
                  animationDuration={500}
                  animationEasing="ease-out"
                  onMouseEnter={(_, i) => setHoveredDelta(i)}
                  onMouseLeave={() => setHoveredDelta(null)}
                  className="cursor-pointer"
                >
                  {rows.map((r, i) => (
                    <Cell
                      key={r.method}
                      fill={colorFor(r.method)}
                      fillOpacity={hoveredDelta === null || hoveredDelta === i ? 1 : 0.35}
                    />
                  ))}
                  <ErrorBar dataKey="errY" stroke="var(--color-lav-text)" strokeWidth={1.4} width={4} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Pre-shift vs. post-shift success rate
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              Fixed Rule starts highest (its blind design is 6/7 correct) but drops hardest and never
              recovers; both Thompson variants dip less and climb back.
            </p>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={rows} margin={{ top: 8, right: 24, left: 0, bottom: 24 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="method"
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />}
                  cursor={{ fill: 'rgba(167,139,250,0.08)' }}
                />
                <Legend
                  content={
                    <InteractiveLegend
                      hidden={shiftSeries.hidden}
                      onToggle={shiftSeries.toggle}
                      onHover={shiftSeries.setHovered}
                    />
                  }
                />
                <Bar
                  dataKey="pre_shift_rate"
                  name="Pre-shift"
                  fill="var(--color-lav-muted)"
                  radius={[6, 6, 0, 0]}
                  fillOpacity={shiftSeries.hoveredKey && shiftSeries.hoveredKey !== 'pre_shift_rate' ? 0.3 : 1}
                  hide={shiftSeries.isHidden('pre_shift_rate')}
                  animationDuration={500} animationEasing="ease-out"
                />
                <Bar
                  dataKey="late_post_shift_rate"
                  name="Late post-shift"
                  fill="var(--color-lav-primary)"
                  radius={[6, 6, 0, 0]}
                  fillOpacity={shiftSeries.hoveredKey && shiftSeries.hoveredKey !== 'late_post_shift_rate' ? 0.3 : 1}
                  hide={shiftSeries.isHidden('late_post_shift_rate')}
                  animationDuration={500} animationEasing="ease-out"
                />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div className="mt-4 mb-1 flex items-center gap-3">
            <div className="h-px flex-1 bg-lav-border" />
            <span className="font-sans text-xs font-semibold uppercase tracking-wide text-lav-muted">
              Task 19B: Drift-Crossover Negative Result
            </span>
            <div className="h-px flex-1 bg-lav-border" />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h3 className="font-sans text-base font-semibold text-lav-text">
                Can a low-dimensional statistic predict whether adaptive learning beats a static table
                under drift?
              </h3>
              <span className="rounded-full border border-lav-cost/40 bg-lav-cost/10 px-3 py-1 font-sans text-xs font-semibold tracking-wide text-lav-cost">
                NEGATIVE RESULT (documented)
              </span>
            </div>
            <p className="font-sans text-sm leading-relaxed text-lav-text">
              In plain terms: this specific attempt at a predictive rule did not work. No formula
              survived testing. It is shown here rather than quietly dropped, because a real
              pattern did survive the process, described below.
            </p>
            <p className="mt-3 font-sans text-sm leading-relaxed text-lav-text">
              The shift above is one drift mechanism. Task 8&apos;s taxonomy (see the Drift Taxonomy
              tab for the full set) covers several mechanistically distinct ones, including
              permutation, monotone decay, partial reorder, and gradual rotation, and Task 19B
              asked whether a low-dimensional summary statistic could predict
              the <em>sign</em> of the adaptive advantage across all four, out of sample, beating a
              majority-class baseline. Three pre-registered attempts were tested, in order; none
              produced a quantitative rule that survives leave-one-cell-out testing. A real, confirmed
              qualitative finding survived the process anyway.
            </p>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Three pre-registered attempts
            </h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              Attempt 2&apos;s apparent R² was largely an artifact of reusing the crossover theory
              under test as an input feature (caught and retracted before being reported as a
              finding, not left standing).
            </p>
            <DataTable
              columns={[
                { key: 'attempt', label: 'Attempt' },
                { key: 'stat', label: 'Sign-prediction result' },
                { key: 'verdict', label: 'Verdict' },
              ]}
              rows={ATTEMPTS}
              sortable
            />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Confirmed finding: reversion, not magnitude, governs the sign
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              R = fraction of argmax-change events where the new optimum had already been optimal
              earlier in that row&apos;s history (&ldquo;does the damage revert?&rdquo;). All 13/13
              negative-advantage cells sit at R ≥ 0.68; all 32/32 R≈0 cells are positive: a
              correlational, mechanism-consistent pattern that held even though no numerical threshold
              built on top of it survived out-of-sample testing.
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={REVERSION_BY_TYPE} margin={{ top: 8, right: 24, left: 0, bottom: 24 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="drift_type"
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  label={{
                    value: 'Reversion fraction (R)',
                    angle: -90,
                    position: 'insideLeft',
                    style: { fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-muted)' },
                  }}
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => v.toFixed(3)} />}
                  cursor={{ fill: 'rgba(167,139,250,0.08)' }}
                />
                <ReferenceLine y={0.68} stroke="var(--color-lav-cost)" strokeDasharray="4 4" label={{ value: 'R = 0.68 (all negative cells above this)', fill: 'var(--color-lav-cost)', fontSize: 11, position: 'insideTopLeft' }} />
                <Bar
                  dataKey="R"
                  name="Reversion fraction"
                  radius={[6, 6, 0, 0]}
                  animationDuration={500}
                  animationEasing="ease-out"
                  onMouseEnter={(_, i) => setHoveredReversion(i)}
                  onMouseLeave={() => setHoveredReversion(null)}
                  className="cursor-pointer"
                >
                  {REVERSION_BY_TYPE.map((r, i) => (
                    <Cell
                      key={r.drift_type}
                      fill={r.negative > 0 ? 'var(--color-lav-cost)' : 'var(--color-lav-success)'}
                      fillOpacity={hoveredReversion === null || hoveredReversion === i ? 1 : 0.35}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric label="Negative-advantage cells" value="13 / 64" accent="cost" help="All sit at R >= 0.68" />
            <Metric label="R≈0 cells" value="32 / 64" accent="success" help="16 monotone_decay + 16 gradual_rotation, all positive" />
            <Metric label="Unaccounted structure" value="+0.092 R²" accent="primary" help="drift_type fixed effects beyond (D, R)" />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <p className="font-sans text-sm leading-relaxed text-lav-text">
              <strong>Why this explains the &ldquo;partial-reorder inversion&rdquo;</strong> (static
              beating adaptive in 10/16 partial_reorder cells): best↔second swaps in this mechanism{' '}
              <em>revert</em>. A static table&apos;s stale action keeps near-optimal value most of the
              time, while an adaptive learner pays real exploration cost chasing a target that keeps
              flipping back to where it already was. Stated as a correlational, mechanism-consistent
              explanation, not a causal claim: no controlled intervention (drift matched on damage
              magnitude, independently varying reversion) was run.
            </p>
            <p className="mt-3 font-sans text-sm leading-relaxed text-lav-text">
              <strong>Explicit scope statement:</strong> the sign of adaptive advantage under drift is{' '}
              <strong>not reducible to a two-statistic summary</strong> across these four
              mechanistically distinct drift regimes, given the data and methods available. Full
              record, including the self-caught circularity in Attempt 2:{' '}
              <span className="font-mono text-lav-muted">docs/theory/drift_crossover_negative.md</span>.
            </p>
          </Card>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
