import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { Play, Pause, RotateCcw, MousePointerClick } from 'lucide-react'
import { Card } from '../../components/ui/Card'
import { cn } from '../../lib/utils'
import { useJson } from '../hooks/useJson'
import { LearningCurveChart } from '../LearningCurveChart'
import { DemoSidePanel } from '../DemoSidePanel'
import { STATIONARY_ANNOTATIONS, DISRUPTED_ANNOTATIONS } from '../data/failureLearnedMapping'
import { RevealText } from '../RevealText'

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
}

export function DemoSection() {
  const { data, loading, error } = useJson('learning_curve.json')
  const [mode, setMode] = useState('stationary')
  const [windowIndex, setWindowIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef(null)

  const windows = data ? (mode === 'stationary' ? data.stationary : data.disrupted) : []
  const maxIndex = windows.length - 1

  useEffect(() => {
    if (!playing || !data) return undefined
    timerRef.current = setInterval(() => {
      setWindowIndex((i) => {
        if (i >= maxIndex) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, 550)
    return () => clearInterval(timerRef.current)
  }, [playing, data, maxIndex])

  const current = windows[windowIndex]
  const annotations = mode === 'stationary' ? STATIONARY_ANNOTATIONS : DISRUPTED_ANNOTATIONS
  const annotation = useMemo(() => {
    if (!current) return ''
    return annotations.find((a) => current.episodeEnd <= a.maxEpisode)?.text ?? annotations[annotations.length - 1].text
  }, [annotations, current])

  const handleModeChange = (next) => {
    setMode(next)
    setWindowIndex(0)
    setPlaying(false)
  }

  return (
    <section id="demo" className="scroll-mt-24 py-28">
      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.5 }}
        variants={fadeUp}
        className="mx-auto max-w-2xl text-center"
      >
        <RevealText as="h2" className="font-heading text-4xl font-semibold text-lav-text sm:text-5xl">
          Watch it learn
        </RevealText>
        <p className="mt-4 font-sans text-base leading-relaxed text-lav-muted">
          This is the real Thompson Sampling run: 500 episodes, averaged across 50 trials. Scrub
          through time, or hit play, and watch the success rate climb as it learns.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.5 }}
        variants={fadeUp}
        className="mx-auto mt-6 flex w-fit items-center gap-2 rounded-full border px-4 py-2"
        style={{ borderColor: 'rgba(52,211,153,0.3)', backgroundColor: 'rgba(52,211,153,0.1)' }}
      >
        <motion.span
          animate={{ x: [0, 3, 0] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
          className="flex"
          style={{ color: 'var(--color-lav-live)' }}
        >
          <MousePointerClick size={16} />
        </motion.span>
        <span className="font-sans text-sm font-medium" style={{ color: 'var(--color-lav-live)' }}>
          Try scrubbing through the curve below, and watch it adapt
        </span>
      </motion.div>

      <motion.div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.1 }}
        variants={fadeUp}
        className="mt-12"
      >
        {loading && <p className="text-center font-sans text-sm text-lav-muted">Loading learning curve…</p>}
        {error && (
          <p className="text-center font-sans text-sm text-lav-cost">Failed to load learning curve data.</p>
        )}

        {current && (
          <>
            <div className="mb-8 flex flex-col items-center gap-2">
              <p className="font-sans text-xs font-medium uppercase tracking-wide text-lav-muted">
                Choose a scenario
              </p>
              <div className="flex flex-wrap items-center justify-center gap-3">
                {[
                  { id: 'stationary', label: 'Stationary' },
                  { id: 'disrupted', label: 'With mid-run disruption (ep. 250)' },
                ].map((opt) => (
                  <button
                    key={opt.id}
                    onClick={() => handleModeChange(opt.id)}
                    className={cn(
                      'rounded-full border-2 px-5 py-2.5 font-sans text-sm font-medium transition-colors',
                      mode === opt.id
                        ? 'text-lav-bg shadow-[0_2px_8px_rgba(52,211,153,0.2)]'
                        : 'border-lav-border bg-lav-surface text-lav-muted hover:text-lav-text',
                    )}
                    style={mode === opt.id ? { borderColor: 'var(--color-lav-live)', backgroundColor: 'var(--color-lav-live)' } : undefined}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_300px]">
              <Card hover={false} className="p-6 sm:p-8">
                <LearningCurveChart
                  windows={windows}
                  currentIndex={windowIndex}
                  shiftEpisode={data.shiftEpisode}
                  showShift={mode === 'disrupted'}
                />

                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <motion.button
                    onClick={() => setPlaying((p) => !p)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-lav-bg"
                    style={{ backgroundColor: 'var(--color-lav-live)' }}
                    aria-label={playing ? 'Pause' : 'Play'}
                  >
                    {playing ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
                  </motion.button>
                  <motion.button
                    onClick={() => { setWindowIndex(0); setPlaying(false) }}
                    whileTap={{ scale: 0.95 }}
                    transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-lav-border text-lav-muted transition-colors hover:text-[var(--color-lav-live)]"
                    aria-label="Reset"
                  >
                    <RotateCcw size={15} />
                  </motion.button>
                  <input
                    type="range"
                    min={0}
                    max={maxIndex}
                    step={1}
                    value={windowIndex}
                    onChange={(e) => { setWindowIndex(Number(e.target.value)); setPlaying(false) }}
                    className="h-1.5 min-w-[80px] flex-1 cursor-pointer appearance-none rounded-full bg-lav-border"
                    style={{ accentColor: 'var(--color-lav-live)' }}
                  />
                  <span className="ml-auto shrink-0 font-mono text-xs text-lav-muted sm:w-28 sm:text-right">
                    ep. {current.episodeStart}-{current.episodeEnd}
                  </span>
                </div>

                <AnimatePresence mode="wait">
                  <motion.p
                    key={annotation}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.25 }}
                    className="mt-5 text-center font-sans text-sm italic"
                    style={{ color: 'var(--color-lav-live)' }}
                  >
                    {annotation}
                  </motion.p>
                </AnimatePresence>
              </Card>

              <DemoSidePanel
                episodesSoFar={current.episodeEnd}
                windowSuccessRate={current.windowSuccessRate}
                cumulativeSuccessRate={current.cumulativeSuccessRate}
              />
            </div>
          </>
        )}
      </motion.div>
    </section>
  )
}
