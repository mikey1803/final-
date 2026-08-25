import { motion } from 'motion/react'
import { FAILURE_TYPES } from '../data/failureTypes'
import { Card } from '../components/ui/Card'

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07, delayChildren: 0.1 } },
}
const item = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
}

export function TaxonomyGrid() {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.3 }}
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
    >
      {FAILURE_TYPES.map((f) => (
        <motion.div key={f.id} variants={item}>
          <Card className="h-full">
            <span
              className="flex h-10 w-10 items-center justify-center rounded-full"
              style={{ backgroundColor: `${f.color}26`, color: f.color }}
            >
              <f.icon size={19} />
            </span>
            <h4 className="mt-3 font-sans text-sm font-semibold text-lav-text">{f.label}</h4>
            <p className="mt-1 font-sans text-xs leading-relaxed text-lav-muted">{f.description}</p>
          </Card>
        </motion.div>
      ))}
    </motion.div>
  )
}
