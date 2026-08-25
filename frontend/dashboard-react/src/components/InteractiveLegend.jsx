import { motion } from 'motion/react'
import { cn } from '../lib/utils'

/**
 * Recharts custom legend content (`<Legend content={<InteractiveLegend .../>} />`)
 * that's actually interactive — click a series to hide/show it (Recharts'
 * own <Legend> only ever renders a static key, nothing wires a click to a
 * `hide` prop), hover a series to dim the others. Styled as pill buttons to
 * match ChartTooltip's surface language.
 */
export function InteractiveLegend({ payload, hidden, onToggle, onHover }) {
  // Recharts' own default <Legend> respects legendType="none" and drops
  // those entries, but handing a custom `content` renderer the payload
  // bypasses that filtering — every band/CI Area (legendType="none",
  // dataKey `${series}__range`) shows up here too unless filtered by hand.
  const visible = payload?.filter((entry) => entry.type !== 'none' && !String(entry.dataKey ?? '').endsWith('__range'))
  if (!visible?.length) return null
  return (
    <div className="mb-1 flex flex-wrap items-center justify-center gap-2">
      {visible.map((entry) => {
        const key = entry.dataKey ?? entry.value
        const isHidden = hidden?.has(key)
        return (
          <motion.button
            key={key}
            type="button"
            onClick={() => onToggle?.(key)}
            onMouseEnter={() => onHover?.(key)}
            onMouseLeave={() => onHover?.(null)}
            whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
            className={cn(
              'flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-sans text-xs transition-colors',
              isHidden ? 'border-lav-border/60 text-lav-muted/50' : 'border-lav-border text-lav-text hover:border-lav-primary/40',
            )}
          >
            <span
              className="h-2 w-2 shrink-0 rounded-full transition-opacity"
              style={{ backgroundColor: entry.color, opacity: isHidden ? 0.3 : 1 }}
            />
            <span className={isHidden ? 'line-through decoration-lav-muted/60' : undefined}>{entry.value}</span>
          </motion.button>
        )
      })}
    </div>
  )
}
