import { motion } from 'motion/react'
import { AnimatedPercent } from './AnimatedPercent'

/** Old->new confidence for one strategy: a resizing bar + a counting %. */
export function BeliefDelta({ from, to }) {
  const fromPct = Math.round(from * 100)
  const toPct = Math.round(to * 100)
  const delta = toPct - fromPct
  return (
    <div className="flex flex-1 items-center gap-3">
      <span className="font-mono text-xs text-lav-muted">{fromPct}%</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-lav-bg">
        <motion.div
          initial={{ width: `${fromPct}%` }}
          animate={{ width: `${toPct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full bg-lav-primary"
        />
      </div>
      <span className="font-mono text-sm font-semibold text-lav-primary">
        <AnimatedPercent from={fromPct} value={toPct} />
      </span>
      {delta !== 0 && (
        <span className={`font-mono text-xs font-medium ${delta > 0 ? 'text-lav-success' : 'text-lav-cost'}`}>
          {delta > 0 ? '+' : ''}
          {delta}
        </span>
      )}
    </div>
  )
}
