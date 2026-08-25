import { PageHeader } from '../components/ui/PageHeader'
import { Metric } from '../components/ui/Metric'
import { Figure } from '../components/ui/Figure'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { useJson } from '../hooks/useJson'

const columns = [
  { key: 'drift_type', label: 'Drift type' },
  { key: 'method', label: 'Method' },
  {
    key: 'is_adaptive',
    label: 'Adaptive',
    align: 'right',
    render: (r) => (r.is_adaptive ? 'Yes' : 'No'),
  },
  {
    key: 'mean_recovery_delta',
    label: 'Recovery delta',
    align: 'right',
    render: (r) => (r.mean_recovery_delta == null ? '-' : r.mean_recovery_delta.toFixed(3)),
  },
  {
    key: 'mean_adaptation_half_life_finite',
    label: 'Half-life (episodes)',
    align: 'right',
    render: (r) => (r.mean_adaptation_half_life_finite == null ? 'Never' : Math.round(r.mean_adaptation_half_life_finite)),
  },
  {
    key: 'fraction_never_recovered',
    label: 'Never recovered',
    align: 'right',
    render: (r) => (r.fraction_never_recovered == null ? '-' : `${(r.fraction_never_recovered * 100).toFixed(1)}%`),
  },
]

export function DriftTaxonomy() {
  const { data: summary, loading, error } = useJson('drift_summary.json')
  const { data: effects } = useJson('drift_effect_sizes.json')

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load drift taxonomy data.</p>

  const driftTypes = new Set(summary.map((r) => r.drift_type))
  const methods = new Set(summary.map((r) => r.method))
  const largeEffects = effects ? effects.filter((r) => r.magnitude === 'large').length : null

  const sortedRows = [...summary].sort((a, b) => a.drift_type.localeCompare(b.drift_type) || b.is_adaptive - a.is_adaptive)

  return (
    <div>
      <PageHeader
        title="Drift Taxonomy"
        subtitle="Task 8/8A: the Non-Stationarity tab shows one worst-case disruption. This is the full set of physically-motivated ways an environment can drift."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <p
            className="border-l-2 pl-4 font-sans text-sm leading-relaxed text-lav-text"
            style={{ borderColor: 'rgba(94,234,212,0.4)' }}
          >
            A single sudden permutation of the reward matrix is the sharpest possible disruption,
            but real tool/API degradation rarely looks like that. This task builds {driftTypes.size}{' '}
            mechanistically distinct drift types (from a one-time permutation to a slow, gradual
            rotation) and re-runs the adaptive-vs-static comparison against each one, across{' '}
            {methods.size} methods.
          </p>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric label="Drift mechanisms" value={driftTypes.size} accent="teal" />
            <Metric label="Methods compared" value={methods.size} accent="teal" />
            {largeEffects !== null && (
              <Metric
                label="Large effect sizes"
                value={`${largeEffects} / ${effects.length}`}
                accent="cost"
                help="Method/drift-type pairs where the adaptive-vs-static difference is a large effect (Cliff's delta)"
              />
            )}
          </div>
        </StaggerItem>

        <StaggerItem>
          <Figure
            src={`${import.meta.env.BASE_URL}figures/drift_taxonomy.png`}
            alt="Overview of the five drift-mechanism taxonomy"
            caption="results/figures/drift_taxonomy.png"
          />
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Figure
              src={`${import.meta.env.BASE_URL}figures/adaptation_curves.png`}
              alt="Adaptation curves across drift mechanisms"
              caption="results/figures/adaptation_curves.png"
            />
            <Figure
              src={`${import.meta.env.BASE_URL}figures/recovery_half_life.png`}
              alt="Recovery half-life across drift mechanisms"
              caption="results/figures/recovery_half_life.png"
            />
          </div>
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Recovery by drift type and method
            </h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              Recovery delta compares success rate late vs. early after the shift; a static rule
              cannot re-learn, so its delta should sit near zero.
            </p>
            <DataTable columns={columns} rows={sortedRows} searchable downloadable sortable filename="drift_taxonomy.csv" />
          </div>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
