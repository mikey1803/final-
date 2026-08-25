import { useEffect } from 'react'
import { motion, useMotionValue, useTransform, animate } from 'motion/react'

/** Counts a % number from `from` to `value`, re-triggering whenever `value` changes. */
export function AnimatedPercent({ from = 0, value }) {
  const mv = useMotionValue(from)
  const rounded = useTransform(mv, (v) => `${Math.round(v)}%`)

  useEffect(() => {
    const controls = animate(mv, value, { duration: 0.8, ease: 'easeOut' })
    return controls.stop
  }, [value, mv])

  return <motion.span>{rounded}</motion.span>
}
