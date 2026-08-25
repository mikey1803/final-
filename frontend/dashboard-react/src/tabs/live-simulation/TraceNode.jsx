import { motion } from 'motion/react'
import { cn } from '../../lib/utils'

const STATUS_STYLES = {
  pending: 'border-lav-border bg-lav-surface text-lav-muted',
  active: 'border-lav-primary bg-lav-primary/10 text-lav-primary',
  done: 'border-lav-primary bg-lav-primary text-lav-bg',
  success: 'border-lav-success bg-lav-success text-lav-bg',
  fail: 'border-lav-cost bg-lav-cost text-lav-bg',
}

/** One node of the hero trace: a spring-in status badge, a label, and an
 * optional one-line mono-font caption that fades in once the node resolves.
 * `size="sm"` is used for the compact retry rows stacked below the primary
 * (attempt 1) trace. */
export function TraceNode({ index, label, status, children, size = 'lg' }) {
  const isTerminal = status === 'success' || status === 'fail'
  const badgeSize = size === 'lg' ? 'h-9 w-9 text-sm' : 'h-6 w-6 text-[10px]'

  return (
    <div className="flex flex-1 flex-col items-center gap-1.5 text-center md:flex-initial md:basis-32">
      <motion.div
        initial={{ scale: 0.7, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 22 }}
        className="relative shrink-0"
      >
        {isTerminal && (
          <motion.span
            initial={{ scale: 1, opacity: 0.6 }}
            animate={{ scale: 1.8, opacity: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className={cn('absolute inset-0 rounded-full', status === 'success' ? 'bg-lav-success' : 'bg-lav-cost')}
          />
        )}
        <span
          className={cn(
            'relative flex shrink-0 items-center justify-center rounded-full border-2 font-mono font-semibold transition-colors duration-300',
            badgeSize,
            STATUS_STYLES[status],
          )}
        >
          {status === 'active' && (
            <motion.span
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
              className="absolute inset-0 rounded-full border-2 border-lav-primary"
            />
          )}
          {index}
        </span>
      </motion.div>

      <p className={cn('font-sans font-medium text-lav-muted', size === 'lg' ? 'text-xs' : 'text-[10px]')}>{label}</p>

      {children && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className={cn('font-mono text-lav-text', size === 'lg' ? 'text-[11px]' : 'text-[10px]')}
        >
          {children}
        </motion.div>
      )}
    </div>
  )
}

/** Connector between two trace nodes — a horizontal line on >=768px, a
 * vertical line below it (the brief's "pipeline spine stacks vertically on
 * mobile" fallback). Fills in with a spring rather than a linear tween, plus
 * a single traveling glow streak (desktop only) — not a repeating pulse. */
export function TraceConnector({ size = 'lg' }) {
  return (
    <div
      className={cn(
        'relative shrink-0 self-center overflow-hidden',
        size === 'lg' ? 'h-6 w-px md:h-px md:w-8 md:flex-1' : 'h-4 w-px md:h-px md:w-5',
      )}
    >
      <motion.div
        initial={{ scaleX: 0, scaleY: 0 }}
        animate={{ scaleX: 1, scaleY: 1 }}
        transition={{ type: 'spring', stiffness: 120, damping: 18 }}
        style={{ transformOrigin: '0 0' }}
        className="absolute inset-0 bg-lav-primary/30"
      />
      <motion.div
        initial={{ x: '-100%' }}
        animate={{ x: '250%' }}
        transition={{ duration: 0.8, ease: 'easeInOut', delay: 0.1 }}
        className="absolute inset-y-0 left-0 hidden w-1/3 bg-gradient-to-r from-transparent via-lav-primary to-transparent md:block"
      />
    </div>
  )
}
