import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { PageHeader } from '../components/ui/PageHeader'
import { Card } from '../components/ui/Card'
import { DataTable } from '../components/ui/DataTable'
import { StaggerGroup, StaggerItem } from '../components/ui/Stagger'
import { cn } from '../lib/utils'
import {
  HEADER,
  STATUS_ROWS,
  CHANNELS,
  KAPPA_BANNER,
  KAPPA_BRANCHES,
  KAPPA_RESULT,
  KAPPA_CALLOUT,
  PER_CELL_ROWS,
  PER_CELL_CAPTION,
  DRIFT_ATTEMPTS,
  DRIFT_CONFIRMED_FINDING,
  SCOPE_LIMITATION,
  DATASET_NOTE,
} from '../data/task19Results'

const PILL_CLASSES = {
  validated: 'border-lav-success/40 bg-lav-success/10 text-lav-success',
  partial: 'border-amber-400/40 bg-amber-400/10 text-amber-400',
  negative: 'border-lav-muted/40 bg-lav-muted/10 text-lav-muted',
}

function StatusPill({ status, color }) {
  return (
    <span
      className={`inline-block whitespace-nowrap rounded-full border px-3 py-1 font-sans text-xs font-semibold tracking-wide ${PILL_CLASSES[color]}`}
    >
      {status}
    </span>
  )
}

const VIEWS = [
  { id: 'overview', label: 'Overview' },
  { id: '19a', label: '19A: Classifier Noise' },
  { id: '19b', label: '19B: Drift Negative Result' },
]

function ViewSwitcher({ view, onChange }) {
  return (
    <div className="mb-6 inline-flex flex-wrap gap-1 rounded-full border border-lav-border bg-lav-surface p-1">
      {VIEWS.map((v) => (
        <button
          key={v.id}
          onClick={() => onChange(v.id)}
          className="relative rounded-full px-3.5 py-1.5 font-sans text-sm font-medium transition-colors"
          style={{ color: view === v.id ? 'var(--color-lav-bg)' : undefined }}
        >
          {view === v.id && (
            <motion.span
              layoutId="task19-view-pill"
              className="absolute inset-0 rounded-full"
              style={{ backgroundColor: 'var(--color-lav-amber)' }}
              transition={{ type: 'spring', stiffness: 260, damping: 28 }}
            />
          )}
          <span className={cn('relative z-10', view !== v.id && 'text-lav-muted')}>{v.label}</span>
        </button>
      ))}
    </div>
  )
}

function OverviewView() {
  return (
    <StaggerGroup className="flex flex-col gap-6">
      <StaggerItem>
        <Card hover={false} className="border-lav-amber/25 bg-lav-amber/[0.03]">
          <p className="font-sans text-sm leading-relaxed text-lav-text">
            This tab is a rigor pass done after the main result, not a new finding on its own.{' '}
            <strong>19A</strong> patches the project's crossover formula (which predicts when
            adaptive learning overtakes a static rule) to account for the fact that the agent only
            ever sees a classifier's guess at the failure type, never the true one, then
            rigorously checks that patch holds on data it was not fit to. <strong>19B</strong>{' '}
            tried to find a simple rule for when a sudden environment change favors a static rule
            vs. adaptive learning. That specific attempt failed: no rule survived testing on
            held-out data. It is reported here as a negative result rather than hidden, alongside
            one real, confirmed pattern that turned up along the way.
          </p>
        </Card>
      </StaggerItem>

      <StaggerItem>
        <div>
          <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">Component status</h3>
          <p className="mb-3 font-sans text-xs text-lav-muted">
            Every component this extension touches, classified independently. A component is
            never rounded up to VALIDATED because a related one is.
          </p>
          <DataTable
            columns={[
              { key: 'component', label: 'Component' },
              { key: 'status', label: 'Status', render: (r) => <StatusPill status={r.status} color={r.statusColor} /> },
              { key: 'justification', label: 'Justification' },
            ]}
            rows={STATUS_ROWS}
            sortable
          />
        </div>
      </StaggerItem>
    </StaggerGroup>
  )
}

function View19A() {
  return (
    <StaggerGroup className="flex flex-col gap-6">
      <StaggerItem>
        <Card hover={false}>
          <h3 className="mb-1 font-sans text-base font-semibold text-lav-text">Two-channel decomposition</h3>
          <p className="mb-4 font-sans text-xs text-lav-muted">
            At the real trained classifier&apos;s measured confusion, mean diagonal accuracy{' '}
            {CHANNELS.meanDiagonalAccuracy}.
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-lav-border bg-lav-bg/40 px-5 py-4">
              <div className="font-sans text-sm text-lav-muted">{CHANNELS.channel1Label}</div>
              <div className="font-mono text-2xl font-semibold" style={{ color: 'var(--color-lav-amber)' }}>
                {CHANNELS.channel1Value}
              </div>
            </div>
            <div className="rounded-xl border border-lav-border bg-lav-bg/40 px-5 py-4">
              <div className="font-sans text-sm text-lav-muted">{CHANNELS.channel2Label}</div>
              <div className="font-mono text-2xl font-semibold" style={{ color: 'var(--color-lav-amber)' }}>
                {CHANNELS.channel2Value}
              </div>
            </div>
          </div>
          <p className="mt-4 font-sans text-sm leading-relaxed text-lav-text">{CHANNELS.caption}</p>
        </Card>
      </StaggerItem>

      <StaggerItem>
        <div className="rounded-lg border px-4 py-2.5 font-sans text-xs text-lav-muted" style={{ borderColor: 'rgba(251,191,103,0.3)', backgroundColor: 'rgba(251,191,103,0.05)' }}>
          {KAPPA_BANNER}
        </div>
      </StaggerItem>

      <StaggerItem>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {[KAPPA_BRANCHES.branchA, KAPPA_BRANCHES.branchB].map((b) => (
            <Card hover={false} key={b.label}>
              <h4 className="mb-3 font-sans text-sm font-semibold text-lav-text">{b.label}</h4>
              <dl className="space-y-2 font-sans text-sm">
                <div className="flex items-center justify-between">
                  <dt className="text-lav-muted">Fitted kappa</dt>
                  <dd className="font-mono text-lav-text">{b.kappa}</dd>
                </div>
                <div className="flex items-center justify-between">
                  <dt className="text-lav-muted">n TRAIN cells</dt>
                  <dd className="font-mono text-lav-text">{b.nTrain}</dd>
                </div>
                <div className="flex items-center justify-between">
                  <dt className="text-lav-muted">Held-out median abs rel error</dt>
                  <dd className="font-mono font-semibold" style={{ color: 'var(--color-lav-amber)' }}>{b.heldOutError}</dd>
                </div>
              </dl>
            </Card>
          ))}
        </div>
      </StaggerItem>

      <StaggerItem>
        <Card hover={false} style={{ borderColor: 'rgba(251,191,103,0.3)', backgroundColor: 'rgba(251,191,103,0.05)' }}>
          <div className="space-y-1.5 font-mono text-sm text-lav-text">
            <p>{KAPPA_RESULT.formula}</p>
            <p className="text-lav-muted">{KAPPA_RESULT.threshold}</p>
            <p className="font-semibold" style={{ color: 'var(--color-lav-amber)' }}>{KAPPA_RESULT.decision}</p>
            <p className="text-lav-muted">{KAPPA_RESULT.bootstrap}</p>
            <p className="text-lav-muted">{KAPPA_RESULT.powerCheck}</p>
          </div>
        </Card>
      </StaggerItem>

      <StaggerItem>
        <div className="rounded-xl border border-amber-400/40 bg-amber-400/5 px-5 py-4">
          <p className="font-sans text-sm leading-relaxed text-lav-text">{KAPPA_CALLOUT}</p>
        </div>
      </StaggerItem>

      <StaggerItem>
        <details className="group">
          <summary className="cursor-pointer select-none font-sans text-sm text-lav-muted transition-colors hover:text-[var(--color-lav-amber)]">
            Show per-cell held-out detail (8 TEST environments)
          </summary>
          <div className="mt-3">
            <DataTable
              columns={[
                { key: 'envId', label: 'env_id' },
                { key: 'ratio', label: 'ratio', align: 'right' },
                { key: 'signedRelError', label: 'signed rel error', align: 'right' },
              ]}
              rows={PER_CELL_ROWS}
              sortable
            />
            <p className="mt-2 font-sans text-xs text-lav-muted">{PER_CELL_CAPTION}</p>
          </div>
        </details>
      </StaggerItem>
    </StaggerGroup>
  )
}

function View19B() {
  return (
    <StaggerGroup className="flex flex-col gap-6">
      <StaggerItem>
        <div>
          <h3 className="mb-3 font-sans text-base font-semibold text-lav-text">Three pre-registered attempts</h3>
          <DataTable
            columns={[
              { key: 'attempt', label: 'Attempt' },
              { key: 'statistic', label: 'Statistic' },
              { key: 'result', label: 'R2 / accuracy' },
              { key: 'verdict', label: 'Verdict' },
            ]}
            rows={DRIFT_ATTEMPTS}
            sortable
          />
        </div>
      </StaggerItem>

      <StaggerItem>
        <div className="rounded-xl border border-lav-success/40 bg-lav-success/5 px-5 py-4">
          <span className="rounded-full border border-lav-success/40 bg-lav-success/10 px-3 py-1 font-sans text-xs font-semibold tracking-wide text-lav-success">
            {DRIFT_CONFIRMED_FINDING.title}
          </span>
          <p className="mt-3 font-sans text-sm font-semibold leading-relaxed text-lav-text">
            {DRIFT_CONFIRMED_FINDING.headline}
          </p>
          <ul className="mt-2 list-inside list-disc space-y-1 font-sans text-sm text-lav-text">
            {DRIFT_CONFIRMED_FINDING.bullets.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        </div>
      </StaggerItem>
    </StaggerGroup>
  )
}

const VIEW_COMPONENTS = { overview: OverviewView, '19a': View19A, '19b': View19B }

export function Task19Theory() {
  const [view, setView] = useState('overview')
  const ActiveView = VIEW_COMPONENTS[view]

  return (
    <div>
      <PageHeader title={HEADER.title} subtitle={HEADER.subtitle} />
      <ViewSwitcher view={view} onChange={setView} />

      <AnimatePresence mode="wait">
        <motion.div
          key={view}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <ActiveView />
        </motion.div>
      </AnimatePresence>

      <div className="mt-8 border-t border-lav-border pt-4">
        <p className="font-sans text-xs leading-relaxed text-lav-muted">{SCOPE_LIMITATION}</p>
        <p className="mt-2 font-mono text-xs text-lav-muted">{DATASET_NOTE}</p>
      </div>
    </div>
  )
}
