import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts'
import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { Metric } from '../components/ui/Metric'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { ChartTooltip } from '../components/ChartTooltip'
import { useJson } from '../hooks/useJson'

const optimalColumns = [
  { key: 'env_id', label: 'Environment' },
  { key: 'quality', label: 'Prior quality (0-7)', align: 'right' },
  {
    key: 'optimal_strength',
    label: 'Optimal strength',
    align: 'right',
    render: (r) => (r.optimal_strength == null ? 'No limit (fully trust it)' : r.optimal_strength.toFixed(1)),
  },
  { key: 'mean_at_optimum', label: 'Score at optimum', align: 'right', render: (r) => r.mean_at_optimum.toFixed(3) },
  { key: 'n_seeds', label: 'Seeds', align: 'right' },
]

function aggregateByQuality(rows) {
  const byQuality = new Map()
  for (const r of rows) {
    if (!byQuality.has(r.quality)) byQuality.set(r.quality, { quality: r.quality, sum: 0, n: 0 })
    const entry = byQuality.get(r.quality)
    entry.sum += r.delta_vs_stock
    entry.n += 1
  }
  return [...byQuality.values()]
    .map((e) => ({ quality: e.quality, meanDelta: e.sum / e.n }))
    .sort((a, b) => a.quality - b.quality)
}

export function PriorGuidedLearning() {
  const { data: optimalStrength, loading, error } = useJson('prior_optimal_strength.json')
  const { data: transferRegion } = useJson('prior_negative_transfer_region.json')
  const [hovered, setHovered] = useState(null)

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load prior-sensitivity data.</p>

  const negativeCount = transferRegion ? transferRegion.filter((r) => r.is_negative_transfer).length : null
  const chartData = transferRegion ? aggregateByQuality(transferRegion) : []
  const qualityLevels = new Set(optimalStrength.map((r) => r.quality)).size

  return (
    <div>
      <PageHeader
        title="Prior-Guided Learning"
        subtitle="Tasks 7/9: starting the adaptive learner from a hand-designed rulebook instead of from scratch, and mapping out exactly when that head start helps vs. hurts."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <p
            className="border-l-2 pl-4 font-sans text-sm leading-relaxed text-lav-text"
            style={{ borderColor: 'rgba(94,234,212,0.4)' }}
          >
            Instead of learning from zero knowledge, Prior-Guided Thompson Sampling (PGTS)
            starts with a hand-designed rulebook as an initial belief, then keeps learning from
            real outcomes on top of it. That head start is not free: if the rulebook is bad, the
            learner has to unlearn it first, which can leave it worse off than starting from
            scratch. The chart below shows exactly that, prior quality against how much it
            changes the outcome compared to a plain Thompson Sampling baseline with no prior.
          </p>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric label="Prior quality levels tested" value={qualityLevels} accent="teal" />
            {negativeCount !== null && (
              <Metric
                label="Configurations where the prior hurt"
                value={`${negativeCount} / ${transferRegion.length}`}
                accent="cost"
                help="Strength/quality/environment combinations where trusting the prior scored below the no-prior baseline"
              />
            )}
            <Metric label="Environments compared" value={new Set(optimalStrength.map((r) => r.env_id)).size} accent="teal" />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Effect on outcome, by prior quality
            </h3>
            <p className="mb-4 font-sans text-xs leading-relaxed text-lav-muted">
              Prior quality: how many of the 7 failure types the injected rulebook guessed
              correctly, 0 (all wrong) to 7 (all correct). Bars below zero mean trusting that
              rulebook did worse than ignoring it entirely.
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 24 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="quality"
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  label={{
                    value: 'Prior quality (0-7)',
                    position: 'bottom',
                    style: { fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-muted)' },
                  }}
                />
                <YAxis
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => v.toFixed(3)} labelFormatter={(v) => `quality = ${v}`} />}
                  cursor={{ fill: 'rgba(94,234,212,0.08)' }}
                />
                <ReferenceLine y={0} stroke="var(--color-lav-text)" />
                <Bar
                  dataKey="meanDelta"
                  name="Mean delta vs. no-prior baseline"
                  radius={[6, 6, 0, 0]}
                  animationDuration={500}
                  animationEasing="ease-out"
                  onMouseEnter={(_, i) => setHovered(i)}
                  onMouseLeave={() => setHovered(null)}
                  className="cursor-pointer"
                >
                  {chartData.map((r, i) => (
                    <Cell
                      key={r.quality}
                      fill={r.meanDelta >= 0 ? 'var(--color-lav-teal)' : 'var(--color-lav-cost)'}
                      fillOpacity={hovered === null || hovered === i ? 1 : 0.35}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Optimal prior strength, per environment and quality level
            </h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              How strongly to trust the prior before it stops paying off. "No limit" means the
              data never found a point where trusting it further hurt.
            </p>
            <DataTable columns={optimalColumns} rows={optimalStrength} downloadable sortable filename="prior_optimal_strength.csv" />
          </div>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
