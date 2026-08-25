import { motion } from 'motion/react'
import { LearningLoopDiagram } from '../LearningLoopDiagram'
import { WhyBetterRecap } from '../WhyBetterRecap'
import { RevealText } from '../RevealText'

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

export function SolutionSection() {
  return (
    <section id="solution" className="scroll-mt-24 py-20">
      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.5 }}
        variants={fadeUp}
        className="mx-auto max-w-2xl text-center"
      >
        <RevealText as="h2" className="font-heading text-4xl font-semibold text-lav-text sm:text-5xl">
          We use online learning
        </RevealText>
        <p className="mt-4 font-sans text-base leading-relaxed text-lav-muted">
          For each failure type, we keep track of which recovery strategies actually work. As new
          failures happen, we try different strategies, learn which one wins, and get better over
          time. This process is called <strong className="text-lav-text">Thompson Sampling</strong>.
          No one hand-writes the answer; the agent finds it through experience.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.2 }}
        variants={fadeUp}
        className="mx-auto mt-12 max-w-2xl"
      >
        <LearningLoopDiagram />
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.4 }}
        variants={fadeUp}
        className="mx-auto mt-20 max-w-2xl text-center"
      >
        <h3 className="font-sans text-xl font-semibold text-lav-text">Why is this better?</h3>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.4 }}
        variants={fadeUp}
        className="mt-6"
      >
        <WhyBetterRecap />
      </motion.div>
    </section>
  )
}
