import { motion } from 'motion/react'
import { RECOVERY_STRATEGIES } from '../../data/recoveryStrategies'

/** The bandit's actual belief state across all 6 strategies, before and
 * after this attempt — not just the chosen one. The API already returns
 * `strategy_scores_before`/`strategy_scores_after` as a full dict over
 * every RecoveryStrategy (see api/main.py's `_confidences_json`); the old
 * UI threw away 5 of those 6 numbers and showed only the winner's single
 * bar. This is the real distribution, real data, nothing synthesized. */
export function StrategyBeliefBars({ before, after, chosen, revealAfter }) {
  return (
    <div className="flex flex-col gap-1.5">
      {RECOVERY_STRATEGIES.map((s) => {
        const b = before?.[s.id] ?? 0.5
        const a = after?.[s.id] ?? b
        const isChosen = s.id === chosen
        const shown = revealAfter ? a : b
        return (
          <div key={s.id} className="flex items-center gap-2.5">
            <span
              className={
                'w-16 shrink-0 text-right font-mono text-[11px] ' +
                (isChosen ? 'font-semibold text-lav-text' : 'text-lav-muted')
              }
            >
              {s.label}
            </span>
            <div className="relative h-[5px] flex-1 overflow-hidden rounded-sm bg-lav-surface-raised">
              {/* Faint marker at the pre-attempt belief, so the shift itself is visible. */}
              <span
                className="absolute top-0 h-full w-px bg-lav-muted/50"
                style={{ left: `${Math.round(b * 100)}%` }}
              />
              <motion.div
                initial={{ width: `${Math.round(b * 100)}%` }}
                animate={{ width: `${Math.round(shown * 100)}%` }}
                transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                className={'h-full rounded-sm ' + (isChosen ? 'bg-lav-primary' : 'bg-lav-muted/40')}
              />
            </div>
            <span
              className={
                'w-9 shrink-0 font-mono text-[11px] tabular-nums ' +
                (isChosen ? 'text-lav-primary' : 'text-lav-muted')
              }
            >
              {Math.round(shown * 100)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
