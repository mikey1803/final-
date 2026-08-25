import { useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ErrorBar,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'
import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { ChartTooltip } from '../components/ChartTooltip'
import { useCsv } from '../hooks/useCsv'
import { colorFor } from '../lib/colors'

export function BaselinesComparison() {
  const { data, loading, error } = useCsv('comparison_summary.csv')
  const [selected, setSelected] = useState(null)
  const [hoveredRate, setHoveredRate] = useState(null)
  const [hoveredNorm, setHoveredNorm] = useState(null)

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load comparison data.</p>

  const rows = [...data]
    .sort((a, b) => b.normalized_late_stage_score - a.normalized_late_stage_score)
    .map((r) => ({
      ...r,
      errY: [
        r.mean_late_stage_success_rate - r.bootstrap_ci95_lower_late_stage,
        r.bootstrap_ci95_upper_late_stage - r.mean_late_stage_success_rate,
      ],
    }))

  const columns = [
    { key: 'method', label: 'Method' },
    {
      key: 'rate',
      label: 'Late-stage success rate',
      align: 'right',
      render: (r) =>
        `${(r.mean_late_stage_success_rate * 100).toFixed(1)}% [${(r.bootstrap_ci95_lower_late_stage * 100).toFixed(1)}, ${(r.bootstrap_ci95_upper_late_stage * 100).toFixed(1)}]`,
    },
    {
      key: 'cost',
      label: 'Avg. recovery cost',
      align: 'right',
      render: (r) => r.mean_average_cost.toFixed(3),
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

  return (
    <div>
      <PageHeader
        title="Baselines Comparison"
        subtitle="Final 11-method comparison, 500 episodes x 50 trials. Normalized score: 0 = UniformRandom floor, 1 = Oracle ceiling."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <Card hover={false}>
            <h3 className="mb-4 font-sans text-base font-semibold text-lav-text">
              Late-stage success rate by method
            </h3>
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 70 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="method"
                  angle={-40}
                  textAnchor="end"
                  interval={0}
                  height={80}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                  tickFormatter={(v) => v.toFixed(1)}
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />}
                  cursor={{ fill: 'rgba(167,139,250,0.08)' }}
                />
                <Bar
                  dataKey="mean_late_stage_success_rate"
                  name="Success rate"
                  radius={[6, 6, 0, 0]}
                  animationDuration={500}
                  animationEasing="ease-out"
                  onMouseEnter={(_, i) => setHoveredRate(i)}
                  onMouseLeave={() => setHoveredRate(null)}
                  onClick={(r) => setSelected(r.method)}
                  className="cursor-pointer"
                >
                  {rows.map((r, i) => (
                    <Cell
                      key={r.method}
                      fill={colorFor(r.method)}
                      fillOpacity={hoveredRate === null || hoveredRate === i ? 1 : 0.35}
                      stroke={selected === r.method ? 'var(--color-lav-text)' : 'none'}
                      strokeWidth={selected === r.method ? 2 : 0}
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
            <h3 className="mb-4 font-sans text-base font-semibold text-lav-text">
              Normalized score (0 = floor, 1 = ceiling)
            </h3>
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 70 }}>
                <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
                <XAxis
                  dataKey="method"
                  angle={-40}
                  textAnchor="end"
                  interval={0}
                  height={80}
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <YAxis
                  tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                  stroke="var(--color-lav-border)"
                />
                <Tooltip
                  content={<ChartTooltip formatter={(v) => v.toFixed(3)} />}
                  cursor={{ fill: 'rgba(167,139,250,0.08)' }}
                />
                <ReferenceLine y={0} stroke="var(--color-lav-muted)" strokeDasharray="3 3" />
                <ReferenceLine y={1} stroke="var(--color-lav-muted)" strokeDasharray="3 3" />
                <Bar
                  dataKey="normalized_late_stage_score"
                  name="Normalized score"
                  radius={[6, 6, 0, 0]}
                  animationDuration={500}
                  animationEasing="ease-out"
                  onMouseEnter={(_, i) => setHoveredNorm(i)}
                  onMouseLeave={() => setHoveredNorm(null)}
                  onClick={(r) => setSelected(r.method)}
                  className="cursor-pointer"
                >
                  {rows.map((r, i) => (
                    <Cell
                      key={r.method}
                      fill={colorFor(r.method)}
                      fillOpacity={hoveredNorm === null || hoveredNorm === i ? 1 : 0.35}
                      stroke={selected === r.method ? 'var(--color-lav-text)' : 'none'}
                      strokeWidth={selected === r.method ? 2 : 0}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <DataTable
            columns={columns}
            rows={rows}
            downloadable
            sortable
            filename="baselines_comparison.csv"
            selectedKey="method"
            selectedValue={selected}
          />
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
