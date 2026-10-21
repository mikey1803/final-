import { motion } from 'motion/react'
import { Target, Bot, XCircle, HelpCircle, ArrowDown } from 'lucide-react'
import { Card } from '../components/ui/Card'

const STEPS = [
  {
    icon: Target,
    tag: 'User asks',
    tone: 'text-lav-text',
    iconBg: 'bg-lav-primary/15 text-lav-primary',
    content: '"What’s the weather in New York?"',
    mono: false,
  },
  {
    icon: Bot,
    tag: 'Agent calls a tool',
    tone: 'text-lav-text',
    iconBg: 'bg-lav-primary/15 text-lav-primary',
    content: 'get_weather(location="NYC, USA")',
    mono: true,
  },
  {
    icon: XCircle,
    tag: 'Tool responds',
    tone: 'text-lav-cost',
    iconBg: 'bg-lav-cost/15 text-lav-cost',
    content: 'Error: invalid location format (expected ISO country code)',
    mono: true,
  },
]

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.16, delayChildren: 0.1 } },
}

const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
}

export function ProblemHero() {
  return (
    <div className="mx-auto w-full max-w-md lg:mx-0 lg:max-w-none">
      <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col items-stretch gap-2">
        {STEPS.map((step, i) => (
          <motion.div key={step.tag} variants={item}>
            <Card hover={false} className="flex items-start gap-3 py-4">
              <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${step.iconBg}`}>
                <step.icon size={17} />
              </span>
              <div className="min-w-0">
                <p className="font-sans text-xs font-medium uppercase tracking-wide text-lav-muted">{step.tag}</p>
                <p className={`mt-0.5 truncate font-sans text-sm ${step.tone} ${step.mono ? 'font-mono' : ''}`}>
                  {step.content}
                </p>
              </div>
            </Card>
            {i < STEPS.length - 1 && (
              <div className="flex justify-center py-1">
                <ArrowDown size={16} className="text-lav-border" />
              </div>
            )}
          </motion.div>
        ))}

        <motion.div variants={item} className="flex justify-center py-1">
          <ArrowDown size={16} className="text-lav-border" />
        </motion.div>

        <motion.div variants={item}>
          <Card
            hover={false}
            className="flex items-start gap-3 border-lav-primary bg-lav-primary/5 py-4 shadow-[0_2px_10px_rgba(76,111,255,0.12)]"
          >
            <motion.span
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.4, ease: 'easeOut', delay: 0.5 }}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-lav-primary text-lav-bg"
            >
              <HelpCircle size={17} />
            </motion.span>
            <div>
              <p className="font-sans text-xs font-medium uppercase tracking-wide text-lav-primary">
                Now what?
              </p>
              <p className="mt-0.5 font-sans text-sm text-lav-text">
                Retry the same call? Give up? Try a different tool? This one decision is the
                difference between an agent that recovers and one that quietly breaks.
              </p>
            </div>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  )
}
