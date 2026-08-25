import { motion } from 'motion/react'
import { ProblemHero } from '../ProblemHero'
import { StuckVsAdaptive } from '../StuckVsAdaptive'
import { TaxonomyGrid } from '../TaxonomyGrid'
import { RevealText } from '../RevealText'

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

const fadeLeft = {
  hidden: { opacity: 0, x: -24 },
  show: { opacity: 1, x: 0, transition: { duration: 0.5, ease: 'easeOut' } },
}

export function ProblemSection() {
  return (
    <section id="problem" className="scroll-mt-24 py-20">
      <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-[1fr_1fr]">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.5 }}
          variants={fadeLeft}
        >
          <RevealText as="h2" className="font-heading text-4xl font-semibold text-lav-text sm:text-5xl">
            Why do LLM agents fail?
          </RevealText>
          <p className="mt-4 max-w-prose font-sans text-base leading-relaxed text-lav-muted">
            Not with a dramatic crash. Usually with one small, silent misstep that nobody planned
            for. Here’s what that actually looks like.
          </p>
        </motion.div>

        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.3 }}
          variants={fadeUp}
        >
          <ProblemHero />
        </motion.div>
      </div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.4 }}
        variants={fadeUp}
        className="mx-auto mt-24 max-w-2xl text-center"
      >
        <RevealText as="h2" className="font-heading text-2xl font-semibold text-lav-text sm:text-3xl">
          Existing solutions are stuck
        </RevealText>
        <p className="mt-3 font-sans text-sm leading-relaxed text-lav-muted">
          The obvious fix is a hand-written rulebook: "if X fails, do Y." But a rulebook can't
          reconsider itself.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.3 }}
        variants={fadeUp}
        className="mx-auto mt-8 max-w-3xl"
      >
        <StuckVsAdaptive />
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.4 }}
        variants={fadeUp}
        className="mx-auto mt-24 max-w-2xl text-center"
      >
        <h2 className="font-heading text-2xl font-semibold text-lav-text sm:text-3xl">
          Seven ways an agent can fail
        </h2>
        <p className="mt-3 font-sans text-sm leading-relaxed text-lav-muted">
          Every failure this project studies falls into one of seven categories: the taxonomy
          the rest of this project is built around.
        </p>
      </motion.div>

      <div className="mt-8">
        <TaxonomyGrid />
      </div>
    </section>
  )
}
