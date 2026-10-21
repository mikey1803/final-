import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { ArrowRight } from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { Metric } from '../components/ui/Metric'
import { Card } from '../components/ui/Card'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { ChartTooltip } from '../components/ChartTooltip'
import { useCsv } from '../hooks/useCsv'
import { TABS } from '../components/DashboardNav'
import { colorFor } from '../lib/colors'

function tab(id) {
  return TABS.find((t) => t.id === id)
}

function goTo(id) {
  window.location.hash = `dashboard/${id}`
}

// Real numbers, not summaries of the sidebar. Each one is independently
// sourced (comparison_summary.csv, nonstationarity_adaptation.csv, the
// matrix_perturbation.py-derived presets behind Try Your Own Scenario, and
// classifier_diagnostics.json) and links straight to the tab that shows the
// underlying evidence, so this reads as an abstract you can click into, not
// a second copy of the nav.
const FINDINGS = [
  {
    tabId: 'baselines',
    stat: '73.6% vs 68.0%',
    headline: 'A blind, hand-designed Fixed Rule beats every adaptive method in a stationary environment.',
    detail: 'It never sees the ground-truth success matrix, yet out-performs Thompson Sampling, UCB1, LinUCB, and every other adaptive baseline this project tested.',
  },
  {
    tabId: 'non-stationarity',
    stat: '76.7% -> 25.1%',
    headline: 'That same Fixed Rule collapses the moment the environment drifts, and barely recovers.',
    detail: 'Thompson Sampling drops far less on the same shift (58.0% -> 49.8%) and keeps recovering the longer it runs — the paper’s central crossover, from real simulation data.',
  },
  {
    tabId: 'matrix-explorer',
    stat: '2 of 3',
    headline: 'The "Fixed Rule wins" ranking is not universal — it depends on the specific success-rate matrix.',
    detail: 'Two of three hand-constructed alternative matrices break the ranking outright. Load them yourself and watch it flip.',
  },
  {
    tabId: 'classifier-diagnostics',
    stat: '93.9% / 64%',
    headline: 'The real trained classifier is strong overall, with one honest, documented weak spot.',
    detail: 'Held-out accuracy across all 7 failure types is 93.9% — but malformed_output recall is only 64%, confused mainly with tool_timeout and tool_error.',
  },
]

const LIVE_TAB_IDS = ['live-simulation', 'beat-the-bandit', 'diagnose', 'matrix-explorer']

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
                    cursor={{ fill: 'rgba(76,111,255,0.06)' }}
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
          <h2 className="mb-1 font-heading text-lg font-semibold text-lav-text">Key findings</h2>
          <p className="mb-4 font-sans text-sm text-lav-muted">
            Four real results, each independently sourced from this project's own data — click through for the evidence.
          </p>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {FINDINGS.map((f) => {
              const t = tab(f.tabId)
              return (
                <button
                  key={f.tabId}
                  onClick={() => goTo(f.tabId)}
                  className="group text-left"
                >
                  <Card className="flex h-full flex-col gap-2 border-l-2 border-l-lav-primary/50">
                    <div className="flex items-center gap-2">
                      {t && <t.icon size={15} className="shrink-0 text-lav-muted" />}
                      <span className="font-mono text-lg font-semibold text-lav-primary">{f.stat}</span>
                    </div>
                    <p className="font-sans text-sm font-medium text-lav-text">{f.headline}</p>
                    <p className="font-sans text-xs leading-relaxed text-lav-muted">{f.detail}</p>
                    <span className="mt-auto flex items-center gap-1 pt-1 font-sans text-xs text-lav-muted transition-colors group-hover:text-lav-primary">
                      See the evidence
                      <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
                    </span>
                  </Card>
                </button>
              )
            })}
          </div>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false} className="mt-6 flex flex-col gap-3 border-dashed sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="font-sans text-sm font-semibold text-lav-text">Don't just read it — try it</h3>
              <p className="mt-0.5 font-sans text-xs text-lav-muted">
                Every one of these runs against the real trained classifier and the real persistent bandit, live.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {LIVE_TAB_IDS.map((id) => {
                const t = tab(id)
                if (!t) return null
                return (
                  <button
                    key={id}
                    onClick={() => goTo(id)}
                    className="flex items-center gap-1.5 rounded-md border border-lav-border bg-lav-surface px-3 py-1.5 font-sans text-xs text-lav-text transition-colors hover:border-lav-primary/50 hover:text-lav-primary"
                  >
                    <t.icon size={13} />
                    {t.label}
                  </button>
                )
              })}
            </div>
          </Card>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
