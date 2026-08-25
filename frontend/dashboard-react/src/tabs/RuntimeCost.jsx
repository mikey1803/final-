import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { Metric } from '../components/ui/Metric'
import { Figure } from '../components/ui/Figure'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { useJson } from '../hooks/useJson'

const costColumns = [
  { key: 'family', label: 'Model family' },
  { key: 'n_samples', label: 'Samples', align: 'right' },
  { key: 'total_tokens', label: 'Total tokens', align: 'right', render: (r) => r.total_tokens.toLocaleString() },
  {
    key: 'total_estimated_cost_usd',
    label: 'Estimated cost',
    align: 'right',
    render: (r) => (r.total_estimated_cost_usd == null ? 'Not exposed by provider' : `$${r.total_estimated_cost_usd.toFixed(4)}`),
  },
]

export function RuntimeCost() {
  const { data: cost, loading, error } = useJson('llm_cost_accounting_summary.json')
  const { data: hardware } = useJson('environment_hardware.json')

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load runtime/cost data.</p>

  const totalTokens = cost.reduce((sum, r) => sum + r.total_tokens, 0)
  const pricedFamilies = cost.filter((r) => r.total_estimated_cost_usd != null)
  const totalCost = pricedFamilies.reduce((sum, r) => sum + r.total_estimated_cost_usd, 0)
  const env = hardware?.[0]

  return (
    <div>
      <PageHeader
        title="Runtime & Cost"
        subtitle="Task 14: real per-decision latency and real (backfilled) API cost accounting, alongside the environment every number in this dashboard was measured on."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <p
            className="border-l-2 pl-4 font-sans text-sm leading-relaxed text-lav-text"
            style={{ borderColor: 'rgba(251,191,103,0.4)' }}
          >
            Accuracy is not the only thing that matters in practice: a recovery strategy that
            works but takes ten times longer, or calls an LLM ten times as often, has a real
            cost. This instruments the actual runtime and token spend behind the LLM-based
            components in this project, not an estimate.
          </p>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric label="Total tokens (all families)" value={totalTokens} accent="amber" />
            <Metric
              label="Estimated cost, priced families"
              value={`$${totalCost.toFixed(4)}`}
              accent="amber"
              help={`${pricedFamilies.length} / ${cost.length} families expose per-token pricing`}
            />
            <Metric label="Families instrumented" value={cost.length} accent="amber" />
          </div>
        </StaggerItem>

        <StaggerItem>
          <Figure
            src={`${import.meta.env.BASE_URL}figures/runtime_quality_tradeoff.png`}
            alt="Runtime and quality tradeoff across methods"
            caption="results/figures/runtime_quality_tradeoff.png"
          />
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">Token and cost accounting by family</h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              One provider does not expose per-token pricing in its API response; stated as such
              rather than estimated.
            </p>
            <DataTable columns={costColumns} rows={cost} sortable />
          </div>
        </StaggerItem>

        {env && (
          <StaggerItem>
            <Card hover={false}>
              <h3 className="mb-3 font-sans text-base font-semibold text-lav-text">
                Measurement environment
              </h3>
              <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
                {[
                  ['Python', `${env.python_version} (${env.python_implementation})`],
                  ['OS', `${env.os} ${env.os_release}`],
                  ['Machine', env.machine],
                  ['Logical CPUs', env.logical_cpu_count],
                ].map(([label, value]) => (
                  <div key={label}>
                    <p className="font-sans text-xs text-lav-muted">{label}</p>
                    <p className="font-mono text-sm text-lav-text">{value}</p>
                  </div>
                ))}
              </div>
            </Card>
          </StaggerItem>
        )}
      </StaggerGroup>
    </div>
  )
}
