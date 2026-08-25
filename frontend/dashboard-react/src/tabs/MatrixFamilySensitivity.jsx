import {
  ComposedChart,
  Area,
  Line,
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
import { Figure } from '../components/ui/Figure'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { ChartTooltip } from '../components/ChartTooltip'
import { InteractiveLegend } from '../components/InteractiveLegend'
import { useCsv } from '../hooks/useCsv'
import { useJson } from '../hooks/useJson'
import { useSeriesToggle } from '../hooks/useSeriesToggle'
import { colorFor } from '../lib/colors'

const ALGORITHMS = ['Adaptive (Thompson)', 'Adaptive (UCB1)']

const tstarColumns = [
  { key: 'scheme', label: 'Perturbation scheme' },
  { key: 'magnitude', label: 'Magnitude' },
  { key: 'n', label: 'n', align: 'right' },
  {
    key: 'fraction_infinite',
    label: 'Never crosses over',
    align: 'right',
    render: (r) => `${(r.fraction_infinite * 100).toFixed(0)}%`,
  },
  {
    key: 'T_star_finite_median',
    label: 'Median T*',
    align: 'right',
    render: (r) => Math.round(r.T_star_finite_median),
  },
  {
    key: 'ci',
    label: '95% CI',
    align: 'right',
    render: (r) =>
      r.T_star_ci95_lower == null || r.T_star_ci95_upper == null
        ? '-'
        : `${Math.round(r.T_star_ci95_lower)} - ${Math.round(r.T_star_ci95_upper)}`,
  },
]

function pivotByGap(rows) {
  const byGap = new Map()
  for (const r of rows) {
    if (!byGap.has(r.gap)) byGap.set(r.gap, { gap: r.gap })
    const entry = byGap.get(r.gap)
    entry[r.algorithm] = r.mean_normalized_score
    entry[`${r.algorithm}__range`] = [r.bootstrap_ci95_lower, r.bootstrap_ci95_upper]
  }
  return [...byGap.values()].sort((a, b) => a.gap - b.gap)
}

export function MatrixFamilySensitivity() {
  const { data, loading, error } = useCsv('matrix_family_summary.csv')
  const { data: tstarData } = useJson('matrix_robustness_tstar.json')
  const series = useSeriesToggle()

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load matrix-family data.</p>

  const rows = pivotByGap(data)
  const n = data[0]?.n ?? 0

  return (
    <div>
      <PageHeader
        title="Matrix-Family Sensitivity"
        subtitle={`Robustness check across 100 randomized reward-matrix instances per gap level x 20 seeds (n=${n} runs per point), instead of relying on the single hand-authored SUCCESS_MATRIX.`}
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Normalized score vs. optimal-arm gap
            </h3>
            <p className="mb-4 font-sans text-xs text-lav-muted">
              Shaded bands = bootstrap 95% CI across the randomized matrix family. Both algorithms
              improve as the gap between the best and next-best strategy widens; Thompson leads UCB1
              at every gap.
            </p>
            <ResponsiveContainer width="100%" height={420}>
              <ComposedChart data={rows} margin={{ top: 8, right: 24, left: 0, bottom: 36 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="gap"
                  type="number"
                  domain={[0.03, 0.32]}
                  ticks={[0.05, 0.1, 0.2, 0.3]}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  label={{
                    value: 'Optimal-arm gap',
                    position: 'bottom',
                    style: { fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-muted)' },
                  }}
                />
                <YAxis
                  domain={[0, 0.6]}
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
                    <ChartTooltip formatter={(v) => v.toFixed(3)} labelFormatter={(v) => `gap = ${v}`} />
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

                {ALGORITHMS.map((a) => (
                  <Area
                    key={`${a}-band`}
                    dataKey={`${a}__range`}
                    stroke="none"
                    fill={colorFor(a)}
                    fillOpacity={series.hoveredKey && series.hoveredKey !== a ? 0.05 : 0.15}
                    hide={series.isHidden(a)}
                    animationDuration={500} animationEasing="ease-out"
                    legendType="none"
                    connectNulls
                  />
                ))}
                {ALGORITHMS.map((a) => (
                  <Line
                    key={a}
                    dataKey={a}
                    name={a}
                    stroke={colorFor(a)}
                    strokeWidth={2.4}
                    strokeOpacity={series.hoveredKey && series.hoveredKey !== a ? 0.3 : 1}
                    dot={{ r: 4, fill: colorFor(a), strokeWidth: 0 }}
                    hide={series.isHidden(a)}
                    animationDuration={500} animationEasing="ease-out"
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
              Show original static figure (includes hand-authored SUCCESS_MATRIX reference points)
            </summary>
            <div className="mt-3">
              <Figure
                src={`${import.meta.env.BASE_URL}figures/matrix_family_sweep.png`}
                alt="Matrix-family sensitivity sweep, original Matplotlib figure"
                caption="results/phase3a/matrix_family_sweep.png (stars mark the hand-authored matrix's score)"
              />
            </div>
          </details>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Does the crossover point itself survive resampling?
            </h3>
            <p className="mb-4 font-sans text-xs leading-relaxed text-lav-muted">
              Beyond the score curve above, two more checks: whether resampled matrices preserve
              the same method ranking, and how much T* (the crossover episode count) itself
              varies once the matrix is no longer hand-authored.
            </p>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Figure
                src={`${import.meta.env.BASE_URL}figures/robustness_ranking_preservation.png`}
                alt="Method ranking preservation across resampled matrices"
                caption="results/figures/robustness_ranking_preservation.png"
              />
              <Figure
                src={`${import.meta.env.BASE_URL}figures/robustness_tstar_distribution.png`}
                alt="Distribution of the crossover point T* across resampled matrices"
                caption="results/figures/robustness_tstar_distribution.png"
              />
            </div>
            {tstarData && (
              <div className="mt-5">
                <DataTable columns={tstarColumns} rows={tstarData} downloadable sortable filename="matrix_robustness_tstar.csv" />
              </div>
            )}
          </Card>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
