import { useCallback, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { Loader2, AlertTriangle, Swords, RotateCcw, CheckCircle2, XCircle, Dices, User, Equal } from 'lucide-react'
import { PageHeader } from '../components/ui/PageHeader'
import { cn } from '../lib/utils'
import { FAILURE_TYPES } from '../data/failureTypes'
import { RECOVERY_STRATEGIES } from '../data/recoveryStrategies'
import { checkHealth, beatBandit } from '../lib/demoApi'

const SCORE_KEY = 'beat-the-bandit-score-v1'
const EMPTY_SCORE = { rounds: 0, human: 0, bandit: 0, ties: 0 }
const REVEAL_DELAY_MS = 500

function loadScore() {
  try {
    const raw = localStorage.getItem(SCORE_KEY)
    return raw ? { ...EMPTY_SCORE, ...JSON.parse(raw) } : EMPTY_SCORE
  } catch {
    return EMPTY_SCORE
  }
}

function saveScore(score) {
  try {
    localStorage.setItem(SCORE_KEY, JSON.stringify(score))
  } catch {
    // scoreboard just won't persist across reloads — not worth surfacing
  }
}

function pickRandomType(excludeId) {
  const pool = excludeId ? FAILURE_TYPES.filter((f) => f.id !== excludeId) : FAILURE_TYPES
  return pool[Math.floor(Math.random() * pool.length)]
}

/** One side of the reveal (you / the bandit): identity, chosen strategy,
 * then the outcome fading in after `revealed` flips — same staged-reveal
 * feel as CompareLivePanel's MethodColumn, simplified to a single stage
 * since the round's data has already arrived by the time this renders. */
function ResultColumn({ label, icon: Icon, tone, strategyLabel, outcome, revealed }) {
  const success = outcome === 'success'
  return (
    <div className="flex flex-col gap-2.5 border-t-2 pt-3" style={{ borderColor: tone }}>
      <div className="flex items-center gap-2">
        <Icon size={14} style={{ color: tone }} className="shrink-0" />
        <p className="font-sans text-xs font-semibold text-lav-text">{label}</p>
      </div>
      <div className="flex min-h-[1.6rem] items-center font-mono text-sm" style={{ color: tone }}>
        {strategyLabel}
      </div>
      <div className="flex min-h-[1.4rem] items-center gap-1.5 font-sans text-xs">
        <AnimatePresence mode="wait">
          {revealed ? (
            <motion.span
              key="outcome"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className={success ? 'flex items-center gap-1.5 font-medium text-lav-success' : 'flex items-center gap-1.5 font-medium text-lav-cost'}
            >
              {success ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
              {success ? 'Success' : 'Failed'}
            </motion.span>
          ) : (
            <motion.span key="pending" className="text-lav-muted">
              resolving...
            </motion.span>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

/** Head-to-head: you pick a recovery strategy blind, the real persistent
 * Thompson selector picks its own for the same failure, both outcomes come
 * from the real hidden SUCCESS_MATRIX via POST /api/demo/beat-the-bandit.
 * The bandit only ever learns from its own pick, never yours — so if you
 * play enough rounds you should watch it visibly get sharper. */
export function BeatTheBandit() {
  const [backendUp, setBackendUp] = useState(null)
  const [currentType, setCurrentType] = useState(() => pickRandomType())
  const [roundNumber, setRoundNumber] = useState(1)
  const [phase, setPhase] = useState('picking') // picking -> loading -> reveal
  const [result, setResult] = useState(null)
  const [revealed, setRevealed] = useState(false)
  const [error, setError] = useState(null)
  const [score, setScore] = useState(loadScore)

  useEffect(() => {
    checkHealth()
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false))
  }, [])

  const handlePick = useCallback(
    async (strategyId) => {
      if (phase !== 'picking') return
      setPhase('loading')
      setError(null)
      setRevealed(false)
      try {
        const data = await beatBandit(currentType.id, strategyId)
        setResult(data)
        setPhase('reveal')
        setScore((prev) => {
          const next = {
            rounds: prev.rounds + 1,
            human: prev.human + (data.winner === 'human' ? 1 : 0),
            bandit: prev.bandit + (data.winner === 'bandit' ? 1 : 0),
            ties: prev.ties + (data.winner === 'tie' ? 1 : 0),
          }
          saveScore(next)
          return next
        })
        setTimeout(() => setRevealed(true), REVEAL_DELAY_MS)
      } catch (e) {
        setError(e.message || 'Round failed')
        setPhase('picking')
      }
    },
    [phase, currentType],
  )

  const handleNextRound = () => {
    setCurrentType((prev) => pickRandomType(prev.id))
    setRoundNumber((n) => n + 1)
    setPhase('picking')
    setResult(null)
    setRevealed(false)
  }

  const handleResetScore = () => {
    setScore(EMPTY_SCORE)
    saveScore(EMPTY_SCORE)
    setRoundNumber(1)
  }

  const humanStrategyLabel = result
    ? (RECOVERY_STRATEGIES.find((s) => s.id === result.human_strategy)?.label ?? result.human_strategy)
    : null
  const banditStrategyLabel = result
    ? (RECOVERY_STRATEGIES.find((s) => s.id === result.bandit_strategy)?.label ?? result.bandit_strategy)
    : null

  return (
    <div>
      <PageHeader
        title="Beat the Bandit"
        subtitle="Pick a recovery strategy blind — you don't know the hidden success-rate matrix any more than the bandit does. Every round, the real persistent Thompson Sampling selector plays its own pick on the same failure and learns from its own real outcome, so watch it get sharper as you play."
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
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-lav-border py-3">
            <div className="flex items-center gap-5 font-mono text-xs text-lav-muted">
              <span>round {score.rounds}</span>
              <span className="text-lav-teal">you {score.human}</span>
              <span className="text-lav-primary">bandit {score.bandit}</span>
              <span>ties {score.ties}</span>
            </div>
            <button
              onClick={handleResetScore}
              className="flex items-center gap-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-text"
            >
              <RotateCcw size={12} /> Reset scoreboard
            </button>
          </div>

          <div className="border-t border-lav-border py-4">
            <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-lav-muted">
              Round {roundNumber} — current failure
            </p>
            <div className="flex items-start gap-2.5 rounded-md border border-lav-border bg-lav-surface px-3 py-2.5">
              <currentType.icon size={16} className="mt-0.5 shrink-0" style={{ color: currentType.color }} />
              <div>
                <p className="font-sans text-sm font-medium text-lav-text">{currentType.label}</p>
                <p className="mt-0.5 font-sans text-xs leading-snug text-lav-muted/80">{currentType.description}</p>
              </div>
            </div>
          </div>

          {phase === 'picking' || phase === 'loading' ? (
            <div className="border-t border-lav-border py-4">
              <p className="mb-2.5 font-mono text-[10px] uppercase tracking-wider text-lav-muted">Pick a recovery strategy</p>
              <div className="grid grid-cols-1 gap-px overflow-hidden rounded-md border border-lav-border bg-lav-border sm:grid-cols-2 lg:grid-cols-3">
                {RECOVERY_STRATEGIES.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => handlePick(s.id)}
                    disabled={phase === 'loading'}
                    className="bg-lav-surface px-3 py-2.5 text-left transition-colors hover:bg-lav-surface-raised/60 disabled:opacity-50"
                  >
                    <span className="block font-sans text-xs font-medium text-lav-text">{s.label}</span>
                    <span className="mt-0.5 block font-sans text-[11px] leading-snug text-lav-muted/70">{s.description}</span>
                  </button>
                ))}
              </div>
              {phase === 'loading' && (
                <p className="mt-3 flex items-center gap-2 font-mono text-xs text-lav-muted">
                  <Loader2 className="animate-spin" size={13} /> calling the real backend...
                </p>
              )}
            </div>
          ) : (
            <div className="border-t border-lav-border py-4">
              <div className="grid grid-cols-2 gap-6">
                <ResultColumn
                  label="You"
                  icon={User}
                  tone="var(--color-lav-teal)"
                  strategyLabel={humanStrategyLabel}
                  outcome={result.human_outcome}
                  revealed={revealed}
                />
                <ResultColumn
                  label="Bandit (Thompson Sampling)"
                  icon={Dices}
                  tone="var(--color-lav-primary)"
                  strategyLabel={banditStrategyLabel}
                  outcome={result.bandit_outcome}
                  revealed={revealed}
                />
              </div>

              <AnimatePresence>
                {revealed && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={cn(
                      'mt-4 flex items-center gap-2 border-l-2 py-1.5 pl-3 font-sans text-sm font-medium',
                      result.winner === 'human'
                        ? 'border-lav-teal text-lav-teal'
                        : result.winner === 'bandit'
                          ? 'border-lav-primary text-lav-primary'
                          : 'border-lav-border text-lav-muted',
                    )}
                  >
                    {result.winner === 'tie' ? <Equal size={15} /> : <Swords size={15} />}
                    {result.winner === 'human'
                      ? 'You win this round'
                      : result.winner === 'bandit'
                        ? 'Bandit wins this round'
                        : 'Tie — both landed the same outcome'}
                  </motion.div>
                )}
              </AnimatePresence>

              {revealed && (
                <motion.button
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  onClick={handleNextRound}
                  whileTap={{ scale: 0.97 }}
                  transition={{ type: 'spring', bounce: 0, duration: 0.25 }}
                  className="mt-4 rounded bg-lav-primary px-4 py-1.5 font-sans text-sm font-medium text-white"
                >
                  Next round
                </motion.button>
              )}
            </div>
          )}

          {error && <div className="border-t border-lav-cost/40 py-3 font-sans text-sm text-lav-cost">{error}</div>}
        </>
      )}
    </div>
  )
}
