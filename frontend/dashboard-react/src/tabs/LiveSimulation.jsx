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
import { Card } from '../components/ui/Card'
import { cn } from '../lib/utils'
import { FAILURE_TYPES } from '../data/failureTypes'
import { useLiveDemo } from './live-simulation/useLiveDemo'
import { LearnedPanel } from './live-simulation/LearnedPanel'
import { CompareLivePanel } from './live-simulation/CompareLivePanel'
import { BatchRunChart } from './live-simulation/BatchRunChart'
import { PipelineView } from './live-simulation/PipelineView'

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut', delay: i * 0.15 } }),
}
const tapSpring = { type: 'spring', bounce: 0, duration: 0.3 }

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
  // re-learn after a shift by construction, Thompson Sampling can. This is
  // the one-click version of that argument instead of two separate buttons
  // the user has to remember to click in order.
  const handleProveIt = async () => {
    if (!demoState?.shift_active) await runShift()
    setBatchUnderDrift(true)
    await runBatch(100)
  }

  const lastBatchRow = batchResult?.rows?.[batchResult.rows.length - 1]
  const hasShiftedSeries = Boolean(batchResult?.shifted_episodes_seen)
  // The honest number: once any shifted-type episodes landed, use their own
  // cumulative rate (not diluted by the 5/7 failure types the shift never
  // touched), same reasoning as the Non-Stationarity tab's shifted-type-only
  // chart. Falls back to the all-types rate when there's no active shift.
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
        <Card hover={false} className="flex items-center gap-2 font-sans text-sm text-lav-muted">
          <Loader2 className="animate-spin" size={16} />
          Checking backend connection...
        </Card>
      )}

      {backendUp === false && (
        <Card hover={false} className="border-lav-cost/40 bg-lav-cost/5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 shrink-0 text-lav-cost" size={20} />
            <div>
              <p className="font-sans text-sm font-semibold text-lav-text">Backend not reachable</p>
              <p className="mt-1 font-sans text-sm text-lav-muted">
                This tab calls a real FastAPI server at{' '}
                <code className="font-mono text-lav-text">localhost:8000</code>. Start it first:
              </p>
              <pre className="mt-2 overflow-x-auto rounded-lg bg-lav-bg px-3 py-2 font-mono text-xs text-lav-text">
                cd backend{'\n'}uvicorn api.main:app --reload --port 8000
              </pre>
            </div>
          </div>
        </Card>
      )}

      {backendUp === true && (
        <>
          {/* Control strip: one horizontal bar for the primary interaction,
              not a tall sidebar of stacked pills eating a third of the page. */}
          <Card hover={false} className="mb-6 flex flex-wrap items-center gap-3 p-3">
            <div className="flex shrink-0 gap-1 rounded-full border border-lav-border p-1">
              <button
                onClick={() => setMode('manual')}
                className={cn(
                  'flex items-center gap-1.5 rounded-full px-3 py-1.5 font-sans text-sm transition-colors',
                  mode === 'manual' ? 'bg-lav-primary text-lav-bg' : 'text-lav-muted hover:text-lav-text',
                )}
              >
                <Dices size={14} /> <span className="hidden sm:inline">Pick a failure type</span>
              </button>
              <button
                onClick={() => setMode('natural')}
                className={cn(
                  'flex items-center gap-1.5 rounded-full px-3 py-1.5 font-sans text-sm transition-colors',
                  mode === 'natural' ? 'bg-lav-primary text-lav-bg' : 'text-lav-muted hover:text-lav-text',
                )}
              >
                <MessageSquare size={14} /> <span className="hidden sm:inline">Type a request</span>
              </button>
            </div>

            <div className="min-w-[220px] flex-1">
              {mode === 'manual' ? (
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full rounded-lg border border-lav-border bg-lav-surface px-3 py-2 font-mono text-sm text-lav-text"
                >
                  {FAILURE_TYPES.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={requestText}
                  onChange={(e) => setRequestText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleNatural()}
                  placeholder="e.g. what's the weather in Tokyo"
                  className="w-full rounded-lg border border-lav-border bg-lav-surface px-3 py-2 font-sans text-sm text-lav-text"
                />
              )}
            </div>

            <motion.button
              onClick={mode === 'manual' ? handleManual : handleNatural}
              disabled={loading || (mode === 'natural' && !requestText.trim())}
              whileHover={loading ? undefined : { scale: 1.04 }}
              whileTap={loading ? undefined : { scale: 0.96 }}
              transition={tapSpring}
              className="shrink-0 rounded-full bg-lav-primary px-5 py-2 font-sans text-sm font-medium text-lav-bg disabled:opacity-40"
            >
              {loading ? (mode === 'manual' ? 'Injecting...' : 'Running...') : mode === 'manual' ? 'Inject' : 'Run'}
            </motion.button>

            <button
              onClick={reset}
              disabled={loading}
              title="Reset demo"
              className="shrink-0 rounded-full border border-lav-border p-2 text-lav-muted transition-colors hover:text-lav-primary disabled:opacity-40"
            >
              <RotateCcw size={15} />
            </button>
          </Card>

          {/* Bento hero row: the live trace is the actual centerpiece (wide
              tile), what the bandit has learned rides alongside it (narrow
              tile) instead of being buried under a wall of buttons. */}
          <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
            <Card hover={false} className="min-h-[420px] p-6 sm:p-8">
              {loading && (
                <div className="flex h-full min-h-[340px] flex-col items-center justify-center gap-3 text-lav-muted">
                  <Loader2 className="animate-spin" size={28} />
                  <p className="font-sans text-sm">Calling the real backend...</p>
                </div>
              )}

              {!loading && !result && (
                <div className="flex h-full min-h-[340px] flex-col items-center justify-center gap-2 text-center text-lav-muted">
                  <p className="font-sans text-sm">No trials yet. Inject a failure to start.</p>
                </div>
              )}

              {!loading && isCleanSuccess && (
                <motion.div
                  initial="hidden"
                  animate="show"
                  custom={0}
                  variants={fadeUp}
                  className="flex min-h-[340px] flex-col items-center justify-center gap-4 text-center"
                >
                  <span className="flex h-14 w-14 items-center justify-center rounded-full bg-lav-success/15 text-lav-success">
                    <CheckCircle2 size={28} />
                  </span>
                  <div>
                    <p className="font-sans text-lg font-semibold text-lav-text">
                      Tool call succeeded - no recovery needed
                    </p>
                    <p className="mt-1 font-mono text-sm text-lav-muted">
                      {result.tool_called}({JSON.stringify(result.tool_args)})
                    </p>
                    <p className="mt-2 font-mono text-xs text-lav-success">-&gt; {JSON.stringify(result.tool_result)}</p>
                  </div>
                </motion.div>
              )}

              {!loading && isRecoveryRun && <PipelineView result={result} />}
            </Card>

            <LearnedPanel demoState={demoState} />
          </div>

          {error && (
            <Card hover={false} className="mb-6 border-lav-cost/40 bg-lav-cost/5 font-sans text-sm text-lav-cost">
              {error}
            </Card>
          )}

          {/* The headline argument as one feature module, not a button in a
              sidebar pointing at a chart three sections away: the CTA, the
              plain-English result, and the chart it produced all live in the
              same place, because they're one causal story. */}
          <Card hover={false} className="mb-6 border-lav-primary/25 p-6 sm:p-8">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <h3 className="font-sans text-lg font-semibold text-lav-text">
                  Does the bandit actually beat the fixed rule?
                </h3>
                <p className="mt-1 max-w-md font-sans text-sm text-lav-muted">
                  Forces a live environment disruption, then races Thompson Sampling against the blind Fixed Rule
                  for 100 fresh episodes to find out.
                </p>
              </div>
              <motion.button
                onClick={handleProveIt}
                disabled={shiftLoading || batchLoading}
                whileHover={shiftLoading || batchLoading ? undefined : { scale: 1.03 }}
                whileTap={shiftLoading || batchLoading ? undefined : { scale: 0.97 }}
                transition={tapSpring}
                className="accent-gradient flex shrink-0 items-center gap-2 rounded-full px-5 py-2.5 font-sans text-sm font-medium text-lav-bg shadow-[0_6px_20px_rgba(0,0,0,0.35)] disabled:opacity-50"
              >
                {shiftLoading || batchLoading ? (
                  <Loader2 className="animate-spin" size={15} />
                ) : (
                  <TrendingUp size={15} />
                )}
                Prove it
              </motion.button>
            </div>

            {demoState?.shift_active && (
              <span className="mt-3 inline-block rounded-full bg-lav-cost/10 px-2.5 py-1 font-mono text-[11px] text-lav-cost">
                shifted: {demoState.shifted_types?.join(', ')}
              </span>
            )}

            {batchResult ? (
              <div className="mt-5">
                {batchGap !== null && (
                  <div
                    className={cn(
                      'mb-3 flex items-center gap-2 rounded-lg px-3.5 py-2.5 font-sans text-sm',
                      batchGap > 0
                        ? 'bg-lav-success/10 text-lav-success'
                        : batchGap < 0
                          ? 'bg-lav-cost/10 text-lav-cost'
                          : 'bg-lav-surface-raised text-lav-muted',
                    )}
                  >
                    <TrendingUp size={16} className="shrink-0" />
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
                <p className="mb-2 font-sans text-xs text-lav-muted">
                  {batchResult.n_episodes} throwaway episodes, simulated server-side (doesn't touch the persistent
                  demo state above).{' '}
                  {hasShiftedSeries
                    ? `${batchResult.shifted_episodes_seen} of them hit a shifted failure type (${batchResult.shifted_types?.join(', ')}). Bold lines below isolate those; dashed lines are the all-types rate, diluted by the 5/7 types the shift never touched.`
                    : batchUnderDrift
                      ? 'Run under an active environment disruption, but no shifted-type episodes happened to land in this batch. Try again.'
                      : 'Run under the normal (non-disrupted) environment.'}
                </p>
                <BatchRunChart rows={batchResult.rows} hasShiftedSeries={hasShiftedSeries} />
              </div>
            ) : (
              <div className="mt-5 flex min-h-[120px] flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-lav-border text-center">
                <p className="font-sans text-sm text-lav-muted">No run yet.</p>
                <p className="font-sans text-xs text-lav-muted/70">Click "Prove it" above to find out.</p>
              </div>
            )}

            {/* Manual controls for the two steps "Prove it" combines, for
                anyone who wants to drive them independently. */}
            <div className="mt-4 flex flex-wrap gap-2 border-t border-lav-border/60 pt-4">
              <button
                onClick={runShift}
                disabled={shiftLoading || demoState?.shift_active}
                className="flex items-center gap-1.5 rounded-full border border-lav-border px-3 py-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-primary disabled:opacity-40"
              >
                {shiftLoading ? <Loader2 className="animate-spin" size={12} /> : <Zap size={12} />}
                Trigger disruption only
              </button>
              <button
                onClick={handleBatch}
                disabled={batchLoading || shiftLoading}
                className="flex items-center gap-1.5 rounded-full border border-lav-border px-3 py-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-primary disabled:opacity-40"
              >
                {batchLoading ? <Loader2 className="animate-spin" size={12} /> : <PlayCircle size={12} />}
                Run 100 episodes only
              </button>
            </div>
          </Card>

          {/* Comparison strip: its trigger lives right next to it now, not in
              a distant sidebar. */}
          <Card hover={false}>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-sans text-sm font-semibold uppercase tracking-wide text-lav-muted">
                Comparison strip: Fixed Rule vs. Thompson vs. LinUCB
              </h3>
              <button
                onClick={handleCompare}
                disabled={mode !== 'manual' || loading || compareLoading}
                title={mode !== 'manual' ? 'Pick a failure type above to compare methods' : undefined}
                className="flex items-center gap-1.5 rounded-full border border-lav-border px-3 py-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-primary disabled:opacity-40"
              >
                {compareLoading ? <Loader2 className="animate-spin" size={13} /> : <Scale size={13} />}
                Compare methods
              </button>
            </div>
            {compareResult ? (
              <CompareLivePanel compareResult={compareResult} />
            ) : (
              <p className="font-sans text-sm text-lav-muted">
                No comparison yet. Pick a failure type above and run "Compare methods".
              </p>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
