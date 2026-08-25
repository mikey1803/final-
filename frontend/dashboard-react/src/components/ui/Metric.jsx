import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useInView, useReducedMotion, animate } from 'motion/react'
import { Card } from './Card'

const ACCENT_CLASSES = {
  primary: 'text-lav-primary',
  success: 'text-lav-success',
  cost: 'text-lav-cost',
  teal: 'text-[var(--color-lav-teal)]',
  amber: 'text-[var(--color-lav-amber)]',
  live: 'text-[var(--color-lav-live)]',
}

/** Parses "47.2%", "13 / 64", "+0.092 R²", "500" etc into a target number to
 * animate toward, plus the prefix/suffix text around it. Falls back to a
 * plain (non-animated) render for values with no leading number, like model
 * family names. */
function parseNumeric(value) {
  const str = String(value)
  const match = str.match(/^([^\d-]*)(-?[\d,]+(?:\.\d+)?)(.*)$/)
  if (!match) return null
  const [, prefix, numStr, suffix] = match
  const num = Number(numStr.replace(/,/g, ''))
  if (Number.isNaN(num)) return null
  const decimals = numStr.includes('.') ? numStr.split('.')[1].length : 0
  return { prefix, num, suffix, decimals }
}

function AnimatedValue({ value }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.6 })
  const reduce = useReducedMotion()
  const parsed = parseNumeric(value)

  useEffect(() => {
    if (!parsed || !inView || !ref.current) return undefined
    if (reduce) {
      ref.current.textContent = `${parsed.prefix}${parsed.num.toFixed(parsed.decimals)}${parsed.suffix}`
      return undefined
    }
    const controls = animate(0, parsed.num, {
      duration: 0.9,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => {
        if (ref.current) ref.current.textContent = `${parsed.prefix}${v.toFixed(parsed.decimals)}${parsed.suffix}`
      },
    })
    return () => controls.stop()
  }, [inView, parsed, reduce])

  // The ref must stay attached to this same span across every render, even
  // before `value` becomes parseable (e.g. while async data is still
  // loading) — Motion's useInView only ever observes whatever ref.current
  // pointed to at mount. Conditionally swapping which element gets the ref
  // (as an earlier version of this component did) means the observer is set
  // up against `null`, `inView` never fires, and the count-up never starts:
  // the tile is stuck showing the literal "0" placeholder forever.
  return <span ref={ref}>{parsed ? `${parsed.prefix}0${parsed.suffix}` : value}</span>
}

/** Hover/focus tooltip for a metric's `help` text — replaces a bare `title`
 * attribute (slow to appear, unstyled, invisible to touch/keyboard users
 * unless they happen to long-press) with a real, keyboard-reachable one
 * that materializes the same way CommandPalette/ChartTooltip do. */
function MetricLabel({ label, help }) {
  const [open, setOpen] = useState(false)

  if (!help) {
    return <div className="font-sans text-sm text-lav-muted">{label}</div>
  }

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="cursor-help border-b border-dotted border-lav-muted/50 font-sans text-sm text-lav-muted transition-colors hover:text-lav-text"
      >
        {label}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            role="tooltip"
            initial={{ opacity: 0, y: 4, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ type: 'spring', bounce: 0, duration: 0.2 }}
            className="absolute bottom-full left-0 z-10 mb-2 w-56 rounded-lg border border-lav-border bg-lav-surface px-3 py-2 font-sans text-xs font-normal leading-relaxed text-lav-text shadow-[0_8px_24px_rgba(0,0,0,0.4)]"
          >
            {help}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** Stat/metric tile matching Streamlit's [data-testid="stMetric"] styling.
 * Numeric-looking values count up from 0 once scrolled into view. */
export function Metric({ label, value, help, accent = 'primary' }) {
  return (
    <Card className="px-5 py-4">
      <MetricLabel label={label} help={help} />
      <div className={`font-mono text-2xl font-semibold ${ACCENT_CLASSES[accent]}`}>
        <AnimatedValue value={value} />
      </div>
    </Card>
  )
}
