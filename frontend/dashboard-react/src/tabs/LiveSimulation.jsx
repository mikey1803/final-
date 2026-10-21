import { useState } from 'react'
import { motion } from 'motion/react'
import {
  Dices,
  MessageSquare,
  RotateCcw,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Scale,
  Zap,
  PlayCircle,
  TrendingUp,
} from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { cn } from '../lib/utils'
import { FAILURE_TYPES } from '../data/failureTypes'
import { useLiveDemo } from './live-simulation/useLiveDemo'
import { LearnedPanel } from './live-simulation/LearnedPanel'
import { CompareLivePanel } from './live-simulation/CompareLivePanel'
import { BatchRunChart } from './live-simulation/BatchRunChart'
import { EventStream } from './live-simulation/EventStream'
import { FailureEventGrid } from './live-simulation/FailureEventGrid'
import { TaskInput } from './live-simulation/TaskInput'

const tapSpring = { type: 'spring', bounce: 0, duration: 0.25 }

export function LiveSimulation() {
  const {
    backendUp,
    loading,
    error,
    result,
    demoState,
    runManual,
    runNatural,
    reset,
    compareLoading,
    compareResult,
    runCompare,
    shiftLoading,
    runShift,
    batchLoading,
    batchResult,
    runBatch,
  } = useLiveDemo()
  const [mode, setMode] = useState('manual')
  const [selectedType, setSelectedType] = useState(FAILURE_TYPES[0].id)
  const [requestText, setRequestText] = useState('')
  const [batchUnderDrift, setBatchUnderDrift] = useState(false)

  const handleManual = () => runManual(selectedType)
  const handleNatural = () => {
    if (!requestText.trim()) return
    runNatural(requestText.trim())
  }
  const handleCompare = () => runCompare(selectedType)
  const handleBatch = () => {
    setBatchUnderDrift(Boolean(demoState?.shift_active))
    runBatch(100)
  }
  // The headline demo: force a mid-run environment disruption (permutes which
  // recovery strategy actually works, same mechanic as the Non-Stationarity
  // experiment), then race 100 fresh episodes through it. Fixed Rule can't
  // re-learn after a shift by construction, Thompson Sampling can.
  const handleProveIt = async () => {
    if (!demoState?.shift_active) await runShift()
    setBatchUnderDrift(true)
    await runBatch(100)
  }

  const lastBatchRow = batchResult?.rows?.[batchResult.rows.length - 1]
  const hasShiftedSeries = Boolean(batchResult?.shifted_episodes_seen)
  const batchGap = lastBatchRow
    ? Math.round(
        hasShiftedSeries
          ? (lastBatchRow.thompson_success_rate_shifted - lastBatchRow.fixed_rule_success_rate_shifted) * 100
          : (lastBatchRow.thompson_success_rate - lastBatchRow.fixed_rule_success_rate) * 100,
      )
    : null

  const isCleanSuccess = result && result.true_failure_type == null && result.outcome === 'success'
  const isRecoveryRun = result && result.true_failure_type != null

  return (
    <div>
      <PageHeader
        title="Live Simulation"
        subtitle="A real interactive demo: pick a failure or type a request, and watch the ACTUAL trained classifier (TF-IDF + LinearSVC) and ACTUAL Thompson Sampling bandit (persistent SQLite state) process it live against a FastAPI backend. Nothing here is pre-scripted."
      />

      {backendUp === null && (
        <div className="flex items-center gap-2 border-t border-lav-border pt-4 font-sans text-sm text-lav-muted">
          <Loader2 className="animate-spin" size={15} />
          Checking backend connection...
        </div>
      )}

      {backendUp === false && (
        <div className="flex items-start gap-3 border-t border-lav-cost/40 pt-4">
          <AlertTriangle className="mt-0.5 shrink-0 text-lav-cost" size={18} />
          <div>
            <p className="font-sans text-sm font-semibold text-lav-text">Backend not reachable</p>
            <p className="mt-1 font-sans text-sm text-lav-muted">
              This tab calls a real FastAPI server at <code className="font-mono text-lav-text">localhost:8000</code>. Start it first:
            </p>
            <pre className="mt-2 overflow-x-auto rounded-md border border-lav-border bg-lav-surface px-3 py-2 font-mono text-xs text-lav-text">
              cd backend{'\n'}uvicorn api.main:app --reload --port 8000
            </pre>
          </div>
        </div>
      )}

      {backendUp === true && (
        <>
          {/* Status line — a readout, not a card. */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-lav-border py-2.5 font-mono text-[11px] text-lav-muted">
            <span className="inline-flex items-center gap-1.5">
              <span className="signal-dot signal-dot--pulse text-lav-success" />
              backend connected
            </span>
            <span>episode {demoState?.episode_number ?? 0}</span>
            {demoState?.shift_active && (
              <span className="text-lav-cost">shifted: {demoState.shifted_types?.join(', ')}</span>
            )}
          </div>

          {/* Event input — the failure grid or a real task submission, not a form. */}
          <div className="border-t border-lav-border py-4">
            <div className="mb-3 flex items-center gap-4">
              <button
                onClick={() => setMode('manual')}
                className={cn(
                  'flex items-center gap-1.5 border-b-2 pb-1.5 font-sans text-xs font-medium transition-colors',
                  mode === 'manual' ? 'border-lav-primary text-lav-text' : 'border-transparent text-lav-muted hover:text-lav-text',
                )}
              >
                <Dices size={13} /> Inject a failure
              </button>
              <button
                onClick={() => setMode('natural')}
                className={cn(
                  'flex items-center gap-1.5 border-b-2 pb-1.5 font-sans text-xs font-medium transition-colors',
                  mode === 'natural' ? 'border-lav-primary text-lav-text' : 'border-transparent text-lav-muted hover:text-lav-text',
                )}
              >
                <MessageSquare size={13} /> Submit a task
              </button>
            </div>

            {mode === 'manual' ? (
              <FailureEventGrid selected={selectedType} onSelect={setSelectedType} />
            ) : (
              <TaskInput value={requestText} onChange={setRequestText} onSubmit={handleNatural} disabled={loading} />
            )}

            <div className="mt-3 flex items-center gap-3">
              <motion.button
                onClick={mode === 'manual' ? handleManual : handleNatural}
                disabled={loading || (mode === 'natural' && !requestText.trim())}
                whileTap={loading ? undefined : { scale: 0.97 }}
                transition={tapSpring}
                className="rounded bg-lav-primary px-4 py-1.5 font-sans text-sm font-medium text-white disabled:opacity-40"
              >
                {loading ? (mode === 'manual' ? 'Injecting...' : 'Running...') : mode === 'manual' ? 'Inject' : 'Run'}
              </motion.button>
              <button
                onClick={reset}
                disabled={loading}
                className="flex items-center gap-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-text disabled:opacity-40"
              >
                <RotateCcw size={12} /> Reset demo
              </button>
            </div>
          </div>

          {/* Live event log — the centerpiece, wide, with what's-learned
              riding alongside it rather than buried under the controls. */}
          <div className="grid grid-cols-1 gap-8 border-t border-lav-border py-5 lg:grid-cols-[1fr_260px]">
            <div className="min-h-[280px]">
              {loading && (
                <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-3 text-lav-muted">
                  <Loader2 className="animate-spin" size={22} />
                  <p className="font-mono text-xs">calling the real backend...</p>
                </div>
              )}

              {!loading && !result && (
                <div className="flex h-full min-h-[240px] flex-col items-center justify-center text-center text-lav-muted">
                  <p className="font-sans text-sm">No trials yet. Inject a failure or submit a task to start.</p>
                </div>
              )}

              {!loading && isCleanSuccess && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex h-full min-h-[240px] flex-col items-center justify-center gap-3 text-center"
                >
                  <CheckCircle2 size={24} className="text-lav-success" />
                  <div>
                    <p className="font-sans text-sm font-semibold text-lav-text">Tool call succeeded, no recovery needed</p>
                    <p className="mt-1 font-mono text-xs text-lav-muted">
                      {result.tool_called}({JSON.stringify(result.tool_args)})
                    </p>
                    <p className="mt-1 font-mono text-xs text-lav-success">-&gt; {JSON.stringify(result.tool_result)}</p>
                  </div>
                </motion.div>
              )}

              {!loading && isRecoveryRun && <EventStream result={result} />}
            </div>

            <LearnedPanel demoState={demoState} />
          </div>

          {error && (
            <div className="border-t border-lav-cost/40 py-3 font-sans text-sm text-lav-cost">{error}</div>
          )}

          {/* The headline argument, one place: the CTA, the plain-English
              result, and the chart it produced, because they're one causal
              story, not three separate UI pieces. */}
          <div className="border-t border-lav-border py-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <h3 className="font-sans text-base font-semibold text-lav-text">Does the bandit beat the fixed rule?</h3>
                <p className="mt-1 max-w-md font-sans text-sm text-lav-muted">
                  Forces a live environment disruption, then races Thompson Sampling against the blind Fixed Rule for
                  100 fresh episodes to find out.
                </p>
              </div>
              <motion.button
                onClick={handleProveIt}
                disabled={shiftLoading || batchLoading}
                whileTap={shiftLoading || batchLoading ? undefined : { scale: 0.97 }}
                transition={tapSpring}
                className="flex shrink-0 items-center gap-2 rounded bg-lav-primary px-4 py-2 font-sans text-sm font-medium text-white disabled:opacity-50"
              >
                {shiftLoading || batchLoading ? <Loader2 className="animate-spin" size={14} /> : <TrendingUp size={14} />}
                Prove it
              </motion.button>
            </div>

            {batchResult ? (
              <div className="mt-5">
                {batchGap !== null && (
                  <div
                    className={cn(
                      'mb-3 flex items-center gap-2 border-l-2 py-1.5 pl-3 font-sans text-sm',
                      batchGap > 0 ? 'border-lav-success text-lav-success' : batchGap < 0 ? 'border-lav-cost text-lav-cost' : 'border-lav-border text-lav-muted',
                    )}
                  >
                    <TrendingUp size={14} className="shrink-0" />
                    <span>
                      {batchGap > 0
                        ? `Thompson Sampling finished ${batchGap} points ahead of Fixed Rule`
                        : batchGap < 0
                          ? `Fixed Rule finished ${Math.abs(batchGap)} points ahead this run`
                          : 'Thompson Sampling and Fixed Rule finished tied this run'}
                      {hasShiftedSeries
                        ? ', on the failure types the disruption actually hit, isolating the recovery signal.'
                        : batchUnderDrift && ', after a live environment disruption Fixed Rule can\'t adapt to.'}
                    </span>
                  </div>
                )}
                <p className="mb-2 font-mono text-[11px] text-lav-muted">
                  {batchResult.n_episodes} throwaway episodes, simulated server-side.{' '}
                  {hasShiftedSeries
                    ? `${batchResult.shifted_episodes_seen} hit a shifted failure type (${batchResult.shifted_types?.join(', ')}). Bold lines isolate those; dashed lines are the all-types rate.`
                    : batchUnderDrift
                      ? 'Run under an active disruption, but no shifted-type episodes landed in this batch. Try again.'
                      : 'Run under the normal (non-disrupted) environment.'}
                </p>
                <BatchRunChart rows={batchResult.rows} hasShiftedSeries={hasShiftedSeries} />
              </div>
            ) : (
              <div className="mt-5 flex min-h-[100px] flex-col items-center justify-center gap-1 border border-dashed border-lav-border text-center">
                <p className="font-sans text-sm text-lav-muted">No run yet.</p>
                <p className="font-sans text-xs text-lav-muted/70">Click "Prove it" above to find out.</p>
              </div>
            )}

            <div className="mt-4 flex flex-wrap gap-4 font-sans text-xs text-lav-muted">
              <button
                onClick={runShift}
                disabled={shiftLoading || demoState?.shift_active}
                className="flex items-center gap-1.5 transition-colors hover:text-lav-text disabled:opacity-40"
              >
                {shiftLoading ? <Loader2 className="animate-spin" size={12} /> : <Zap size={12} />}
                Trigger disruption only
              </button>
              <button
                onClick={handleBatch}
                disabled={batchLoading || shiftLoading}
                className="flex items-center gap-1.5 transition-colors hover:text-lav-text disabled:opacity-40"
              >
                {batchLoading ? <Loader2 className="animate-spin" size={12} /> : <PlayCircle size={12} />}
                Run 100 episodes only
              </button>
            </div>
          </div>

          {/* Comparison strip */}
          <div className="border-t border-lav-border py-5">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-mono text-[10px] uppercase tracking-wider text-lav-muted">
                Fixed Rule vs. Thompson vs. LinUCB vs. Naive Retry
              </h3>
              <button
                onClick={handleCompare}
                disabled={mode !== 'manual' || loading || compareLoading}
                title={mode !== 'manual' ? 'Pick a failure event above to compare methods' : undefined}
                className="flex items-center gap-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-text disabled:opacity-40"
              >
                {compareLoading ? <Loader2 className="animate-spin" size={13} /> : <Scale size={13} />}
                Compare methods
              </button>
            </div>
            {compareResult ? (
              <CompareLivePanel compareResult={compareResult} />
            ) : (
              <p className="mt-3 font-sans text-sm text-lav-muted">
                No comparison yet. Pick a failure event above and run "Compare methods".
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
