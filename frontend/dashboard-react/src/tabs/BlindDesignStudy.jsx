import { AlertTriangle } from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { Metric } from '../components/ui/Metric'
import { Figure } from '../components/ui/Figure'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { useJson } from '../hooks/useJson'

const columns = [
  { key: 'family', label: 'Model family' },
  { key: 'n_parsed', label: 'Designs collected', align: 'right' },
  {
    key: 'n_distinct_mappings',
    label: 'Distinct rulebooks',
    align: 'right',
    render: (r) => `${r.n_distinct_mappings} / ${r.n_parsed}`,
  },
  { key: 'q_mean', label: 'Q score (mean)', align: 'right', render: (r) => r.q_mean.toFixed(2) },
  { key: 'q_median', label: 'Q score (median)', align: 'right', render: (r) => r.q_median.toFixed(1) },
]

export function BlindDesignStudy() {
  const { data, loading, error } = useJson('blind_design_by_family.json')

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load blind-design study data.</p>

  const totalDesigns = data.reduce((sum, r) => sum + r.n_parsed, 0)
  const mostConsistent = [...data].sort((a, b) => a.n_distinct_mappings / a.n_parsed - b.n_distinct_mappings / b.n_parsed)[0]

  return (
    <div>
      <PageHeader
        title="Blind-Design Study"
        subtitle="Task 11: is the project's headline result an artifact of one person's hand-designed rulebook, or does it hold up against independent designs?"
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <p
            className="border-l-2 pl-4 font-sans text-sm leading-relaxed text-lav-text"
            style={{ borderColor: 'rgba(251,191,103,0.4)' }}
          >
            The static "Fixed Rule" baseline used everywhere else in this dashboard is one
            person's best guess at good recovery defaults, made without ever being shown which
            strategy actually works best. This study asks three real LLM families to
            independently design their own blind rulebook under the same no-feedback
            constraint, to check whether that one guess was representative rather than lucky.
          </p>
        </StaggerItem>

        <StaggerItem>
          <Card hover={false} className="flex items-start gap-3 border-lav-cost/40 bg-lav-cost/5">
            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-lav-cost" />
            <p className="font-sans text-sm text-lav-text">
              <strong>Human arm: not yet run.</strong> A matching study asking human designers is
              planned but pending ethics approval, so it currently stands at n=0. Stated here
              plainly rather than left out.
            </p>
          </Card>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric label="Model families" value={data.length} accent="amber" />
            <Metric label="Total designs collected" value={totalDesigns} accent="amber" />
            {mostConsistent && (
              <Metric
                label="Most consistent family"
                value={mostConsistent.family}
                accent="amber"
                help={`${mostConsistent.n_distinct_mappings} distinct rulebooks across ${mostConsistent.n_parsed} designs`}
              />
            )}
          </div>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Figure
              src={`${import.meta.env.BASE_URL}figures/llm_study_per_type_accuracy_by_family.png`}
              alt="Per-failure-type design accuracy by model family"
              caption="results/figures/llm_study_per_type_accuracy_by_family.png"
            />
            <Figure
              src={`${import.meta.env.BASE_URL}figures/llm_study_q_by_family.png`}
              alt="Design quality score by model family"
              caption="results/figures/llm_study_q_by_family.png"
            />
          </div>
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">Designs by family</h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              "Distinct rulebooks" counts how many different failure-to-strategy mappings each
              family produced across its repeated, independent attempts. A family that mostly
              repeats one mapping is behaving consistently; one that varies widely is not.
            </p>
            <DataTable columns={columns} rows={data} sortable />
          </div>
        </StaggerItem>
      </StaggerGroup>
    </div>
  )
}
