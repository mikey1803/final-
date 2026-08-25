import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { Metric } from '../components/ui/Metric'
import { Figure } from '../components/ui/Figure'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { useJson } from '../hooks/useJson'

const FIGURES = [
  { src: 'crossover_calibration_curve.png', alt: 'Calibration curve for the analytic crossover predictor' },
  { src: 'crossover_prediction_scatter.png', alt: 'Predicted vs. measured crossover point, scatter plot' },
  { src: 'crossover_error_histogram.png', alt: 'Histogram of prediction error across calibration groups' },
  { src: 'crossover_reliability_heatmap.png', alt: 'Reliability heatmap of the calibrated predictor' },
]

const columns = [
  { key: 'group', label: 'Group' },
  { key: 'n', label: 'n', align: 'right' },
  { key: 'median', label: 'Median T*', align: 'right', render: (r) => r.median.toFixed(1) },
  {
    key: 'within_factor2',
    label: 'Within 2x',
    align: 'right',
    render: (r) => `${(r.within_factor2 * 100).toFixed(0)}%`,
  },
  {
    key: 'within_factor3',
    label: 'Within 3x',
    align: 'right',
    render: (r) => `${(r.within_factor3 * 100).toFixed(0)}%`,
  },
  { key: 'best_fit_c', label: 'Best-fit constant', align: 'right', render: (r) => r.best_fit_c.toFixed(3) },
  {
    key: 'calibrated',
    label: 'Calibrated',
    align: 'right',
    render: (r) => (
      <span className={r.calibrated ? 'text-lav-success' : 'text-lav-muted'}>{r.calibrated ? 'Yes' : 'No'}</span>
    ),
  },
]

const horizonColumns = [
  { key: 'env_id', label: 'Environment' },
  { key: 'classifier_id', label: 'Classifier' },
  { key: 'n_seeds', label: 'Seeds', align: 'right' },
  { key: 'median', label: 'Median T*', align: 'right', render: (r) => Math.round(r.median) },
  {
    key: 'iqr',
    label: 'IQR',
    align: 'right',
    render: (r) => `${Math.round(r.iqr_low)} - ${Math.round(r.iqr_high)}`,
  },
  {
    key: 'fraction_never_crossed',
    label: 'Never crosses over',
    align: 'right',
    render: (r) => `${(r.fraction_never_crossed * 100).toFixed(0)}%`,
  },
]

export function CrossoverPredictor() {
  const { data, loading, error } = useJson('crossover_calibration.json')
  const { data: horizonData } = useJson('horizon_crossover.json')

  if (loading) return <p className="font-sans text-sm text-lav-muted">Loading…</p>
  if (error) return <p className="font-sans text-sm text-red-500">Failed to load crossover calibration data.</p>

  const total = data.length
  const calibratedCount = data.filter((r) => r.calibrated).length
  const validatedCount = data.filter((r) => r.validated_region).length
  const meanWithinFactor2 = data.reduce((sum, r) => sum + r.within_factor2, 0) / total

  const horizonAsymptotic = horizonData ? horizonData.filter((r) => r.criterion === 'asymptotic') : []
  const neverCrossedCount = horizonAsymptotic.filter((r) => r.fraction_never_crossed > 0).length

  return (
    <div>
      <PageHeader
        title="Crossover Predictor"
        subtitle="Task 10/10B: a calibrated formula for the point where adaptive learning overtakes a static rulebook, checked against direct simulation."
      />

      <StaggerGroup className="flex flex-col gap-6">
        <StaggerItem>
          <p
            className="border-l-2 pl-4 font-sans text-sm leading-relaxed text-lav-text"
            style={{ borderColor: 'rgba(94,234,212,0.4)' }}
          >
            Earlier results in this dashboard measure the crossover point (T*, the number of
            episodes before adaptive learning pulls ahead of a static rule) by running the full
            simulation and reading it off a curve. This task asks whether T* can instead be{' '}
            <em>predicted</em> directly from a closed-form formula, calibrated against the
            measured data and validated on groups it was not fit to.
          </p>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric
              label="Calibration groups"
              value={`${calibratedCount} / ${total}`}
              accent="teal"
              help="Groups where the predictor's constant was successfully calibrated"
            />
            <Metric
              label="Validated region"
              value={`${validatedCount} / ${total}`}
              accent="teal"
              help="Groups where the calibrated predictor holds on held-out data"
            />
            <Metric
              label="Mean accuracy, within 2x"
              value={`${(meanWithinFactor2 * 100).toFixed(0)}%`}
              accent="cost"
              help="Average fraction of predictions landing within 2x of the measured T*"
            />
          </div>
        </StaggerItem>

        <StaggerItem>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {FIGURES.map((f) => (
              <Figure
                key={f.src}
                src={`${import.meta.env.BASE_URL}figures/${f.src}`}
                alt={f.alt}
                caption={`results/figures/${f.src}`}
              />
            ))}
          </div>
        </StaggerItem>

        <StaggerItem>
          <div>
            <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
              Per-group calibration detail
            </h3>
            <p className="mb-3 font-sans text-xs text-lav-muted">
              Each group is a distinct combination of environment conditions the predictor was
              tested against, calibrated and validated independently.
            </p>
            <DataTable columns={columns} rows={data} downloadable sortable filename="crossover_calibration.csv" />
          </div>
        </StaggerItem>

        {horizonAsymptotic.length > 0 && (
          <StaggerItem>
            <Card hover={false}>
              <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">
                Does the crossover point hold across different interaction budgets?
              </h3>
              <p className="mb-4 font-sans text-xs leading-relaxed text-lav-muted">
                Task 6: the same measurement repeated at 56 environment/classifier combinations,
                checking whether Thompson Sampling reliably overtakes the static rule regardless
                of how long the agent gets to run, not just at the 500-episode horizon used
                elsewhere in this dashboard.
              </p>
              <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Metric label="Environments tested" value={horizonAsymptotic.length} accent="teal" />
                <Metric
                  label="Never crosses over (some seeds)"
                  value={`${neverCrossedCount} / ${horizonAsymptotic.length}`}
                  accent="cost"
                  help="Environments where at least some seeds never saw Thompson Sampling overtake the static rule"
                />
              </div>
              <details className="group">
                <summary className="cursor-pointer select-none font-sans text-sm text-lav-muted transition-colors hover:text-[var(--color-lav-teal)]">
                  Show all {horizonAsymptotic.length} environment/classifier results
                </summary>
                <div className="mt-3">
                  <DataTable columns={horizonColumns} rows={horizonAsymptotic} searchable downloadable sortable filename="horizon_crossover.csv" />
                </div>
              </details>
            </Card>
          </StaggerItem>
        )}
      </StaggerGroup>
    </div>
  )
}
