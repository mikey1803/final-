import { motion } from 'motion/react'
import { cn } from '../../lib/utils'

/**
 * Base surface card (dark lavender void surface, hairline border, soft
 * shadow), with a restrained Motion hover lift (y-shift + border-color
 * shift, no scale jump).
 */
export function Card({ className, children, hover = true, ...props }) {
  return (
    <motion.div
      className={cn(
        'rounded-xl border border-lav-border bg-lav-surface p-6 shadow-[0_2px_8px_rgba(0,0,0,0.20)]',
        className,
      )}
      whileHover={
        hover
          ? {
              y: -2,
              boxShadow: '0 8px 20px rgba(0,0,0,0.30)',
              borderColor: 'rgba(167,139,250,0.28)',
            }
          : undefined
      }
      transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
      {...props}
    >
      {children}
    </motion.div>
  )
}
