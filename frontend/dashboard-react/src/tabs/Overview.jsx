import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { PageHeader } from '../components/ui/PageHeader'
import { Metric } from '../components/ui/Metric'
import { Card } from '../components/ui/Card'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { ChartTooltip } from '../components/ChartTooltip'
import { useCsv } from '../hooks/useCsv'
import { NAV_GROUPS, groupColorFor } from '../components/DashboardNav'
import { colorFor } from '../lib/colors'

// Derived from the sidebar's own tab list (DashboardNav.jsx) rather than a
// separate hardcoded array, so this page's summaries and the sidebar's
// summaries can never drift apart.
const SECTIONS = NAV_GROUPS.flatMap((g) => g.tabs).filter((t) => t.id !== 'overview')

export function Overview() {
  const { data } = useCsv('comparison_summary.csv')
  const [hovered, setHovered] = useState(null)

  const byMethod = Object.fromEntries((data ?? []).map((r) => [r.method, r]))
  const fmt = (v) => (v === undefined ? '-' : `${(v * 100).toFixed(1)}%`)
  const chartRows = data ? [...data].sort((a, b) => b.normalized_late_stage_score - a.normalized_late_stage_score) : []

  return (
    <div>
      <PageHeader
        title="Failure-Aware Adaptive Recovery Framework"
        subtitle="Multi-armed bandit recovery-strategy selection for LLM agent tool failures. Phase 3-4 results, 500 episodes x 50 trials."
      />

      <StaggerGroup className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StaggerItem>
          <Metric label="Oracle ceiling" value={fmt(byMethod.Oracle?.mean_late_stage_success_rate)} accent="primary" />
        </StaggerItem>
        <StaggerItem>
          <Metric
            label="Best baseline (Fixed Rule, blind)"
            value={fmt(byMethod['Fixed Rule']?.mean_late_stage_success_rate)}
            accent="success"
          />
        </StaggerItem>
        <StaggerItem>
          <Metric
            label="Best adaptive (Thompson)"
            value={fmt(byMethod['Adaptive (Thompson)']?.mean_late_stage_success_rate)}
            accent="primary"
          />
        </StaggerItem>
        <StaggerItem>
          <Metric label="Methods compared" value="11" accent="cost" />
        </StaggerItem>
      </StaggerGroup>

      <StaggerGroup className="mt-6">
        <StaggerItem>
          <p className="mb-6 border-l-2 border-lav-primary/40 pl-4 font-sans text-sm leading-relaxed text-lav-text">
            This project presents a failure-aware adaptive reasoning framework for LLM agents: a
            multi-armed bandit strategy selector (Thompson Sampling, UCB1, epsilon-greedy variants,
            a Contextual Cost-Aware selector, and a plain LinUCB baseline) that recovers from tool
            execution failures. The key finding across this dashboard: a{' '}
            <strong>blind, hand-designed Fixed Rule</strong> baseline (designed with zero access to
            the ground-truth reward matrix) beats every adaptive method in a stationary environment,
            but cannot adapt when the environment shifts mid-run. The tabs below unpack that
            result, and the wider body of work built on top of it, from every angle.
          </p>
        </StaggerItem>

        {chartRows.length > 0 && (
          <StaggerItem>
            <Card hover={false} className="mb-6">
              <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
                All 11 methods, normalized score
              </h3>
              <p className="mb-4 font-sans text-xs text-lav-muted">
                0 = UniformRandom floor, 1 = Oracle ceiling. Stationary environment, 500 episodes x
                50 trials.
              </p>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={chartRows} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
                  <CartesianGrid stroke="var(--color-lav-border)" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
                    stroke="var(--color-lav-border)"
                    tickFormatter={(v) => v.toFixed(1)}
                  />
                  <YAxis
                    type="category"
                    dataKey="method"
                    width={168}
                    tick={{ fontFamily: 'Inter, sans-serif', fontSize: 12, fill: 'var(--color-lav-text)' }}
                    stroke="var(--color-lav-border)"
                  />
                  <Tooltip
                    content={<ChartTooltip formatter={(v) => v.toFixed(3)} />}
                    cursor={{ fill: 'rgba(167,139,250,0.06)' }}
                  />
                  <Bar
                    dataKey="normalized_late_stage_score"
                    name="Normalized score"
                    radius={[0, 6, 6, 0]}
                    animationDuration={500}
                    animationEasing="ease-out"
                    onMouseEnter={(_, i) => setHovered(i)}
                    onMouseLeave={() => setHovered(null)}
                    className="cursor-pointer"
                  >
                    {chartRows.map((r, i) => (
                      <Cell
                        key={r.method}
                        fill={colorFor(r.method)}
                        fillOpacity={hovered === null || hovered === i ? 1 : 0.35}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </StaggerItem>
        )}

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {SECTIONS.map((s) => (
              <Card key={s.id} className="flex items-start gap-3">
                <s.icon className="mt-0.5 shrink-0" size={20} style={{ color: groupColorFor(s.id) }} />
                <div>
                  <h3 className="font-sans text-base font-semibold text-lav-text">{s.label}</h3>
                  <p className="mt-1 font-sans text-sm leading-relaxed text-lav-muted">{s.description}</p>
                </div>
              </Card>
            ))}
          </div>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
