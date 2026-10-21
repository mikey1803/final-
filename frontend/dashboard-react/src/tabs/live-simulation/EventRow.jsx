import { motion } from 'motion/react'
import { cn } from '../../lib/utils'

const TONE_COLOR = {
  neutral: 'var(--color-lav-muted)',
  detect: 'var(--color-lav-amber)',
  live: 'var(--color-lav-live)',
  adaptive: 'var(--color-lav-primary)',
  success: 'var(--color-lav-success)',
  fail: 'var(--color-lav-cost)',
}

/** One row of the live event log — a vertical instrument trace instead of
 * a horizontal stepper of numbered circles. `status` drives the dot:
 * pending (hollow, dim), active (pulsing, tone-colored), done (filled,
 * tone-colored). The connecting spine is a single continuous line drawn
 * by the parent (see EventStream), not a separate connector between each
 * pair of nodes. */
export function EventRow({ index, label, status = 'done', tone = 'neutral', children, compact = false }) {
  const color = TONE_COLOR[tone]
  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={cn('relative flex gap-3', compact ? 'py-1.5' : 'py-2.5')}
    >
      <span className="relative z-10 mt-[3px] flex h-3 w-3 shrink-0 items-center justify-center">
        {status === 'active' && (
          <motion.span
            animate={{ scale: [1, 2.2], opacity: [0.5, 0] }}
            transition={{ duration: 1.1, repeat: Infinity, ease: 'easeOut' }}
            className="absolute h-2 w-2 rounded-full"
            style={{ backgroundColor: color }}
          />
        )}
        <span
          className="relative h-2 w-2 rounded-full border"
          style={{
            borderColor: status === 'pending' ? 'var(--color-lav-border)' : color,
            backgroundColor: status === 'pending' ? 'transparent' : color,
          }}
        />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          {index !== undefined && (
            <span className="font-mono text-[10px] tabular-nums text-lav-muted/70">
              {typeof index === 'number' ? String(index).padStart(2, '0') : index}
            </span>
          )}
          <span className={cn('font-mono text-[10px] uppercase tracking-wider', status === 'pending' ? 'text-lav-muted/50' : 'text-lav-muted')}>
            {label}
          </span>
        </div>
        {children && <div className="mt-1 font-sans text-sm text-lav-text">{children}</div>}
      </div>
    </motion.div>
  )
}

/** The continuous vertical spine every EventRow sits on. Wraps a list of
 * rows so the connecting line is one element, not N-1 separate connectors. */
export function EventSpine({ children }) {
  return (
    <div className="relative">
      <div className="absolute left-[5px] top-2 bottom-2 w-px bg-lav-border" aria-hidden="true" />
      {children}
    </div>
  )
}
