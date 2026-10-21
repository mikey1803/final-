import { motion } from 'motion/react'
import { cn } from '../../lib/utils'

/**
 * Base surface panel: hairline border, flat (no ambient shadow — shadows
 * are reserved for things that actually float above content, like the
 * command palette, not every panel on the page). Radius kept modest
 * rather than the soft/bubbly `rounded-xl` this system used to default to
 * everywhere, so panels read as structure, not as decoration.
 */
export function Card({ className, children, hover = true, ...props }) {
  return (
    <motion.div
      className={cn('rounded-lg border border-lav-border bg-lav-surface p-6', className)}
      whileHover={
        hover
          ? {
              borderColor: 'rgba(76,111,255,0.32)',
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
