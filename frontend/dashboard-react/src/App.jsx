import { useEffect, useState } from 'react'
import { AnimatePresence, motion, MotionConfig } from 'motion/react'
import { DashboardTopBar, DashboardSidebar } from './components/DashboardNav'
import { CommandPalette } from './components/CommandPalette'
import { Overview } from './tabs/Overview'
import { ClassifierNoiseSweep } from './tabs/ClassifierNoiseSweep'
import { MatrixFamilySensitivity } from './tabs/MatrixFamilySensitivity'
import { NonStationarity } from './tabs/NonStationarity'
import { BaselinesComparison } from './tabs/BaselinesComparison'
import { LiveSimulation } from './tabs/LiveSimulation'
import { Task19Theory } from './tabs/Task19Theory'
import { CrossoverPredictor } from './tabs/CrossoverPredictor'
import { DriftTaxonomy } from './tabs/DriftTaxonomy'
import { BlindDesignStudy } from './tabs/BlindDesignStudy'
import { PriorGuidedLearning } from './tabs/PriorGuidedLearning'
import { RuntimeCost } from './tabs/RuntimeCost'
import { MethodologyIntegrity } from './tabs/MethodologyIntegrity'
import { StoryPage } from './story/StoryPage'
import { useScrollGradient } from './hooks/useScrollGradient'

const TAB_COMPONENTS = {
  overview: Overview,
  'classifier-noise': ClassifierNoiseSweep,
  'matrix-family': MatrixFamilySensitivity,
  'non-stationarity': NonStationarity,
  'live-simulation': LiveSimulation,
  baselines: BaselinesComparison,
  'theory-19': Task19Theory,
  'crossover-predictor': CrossoverPredictor,
  'drift-taxonomy': DriftTaxonomy,
  'blind-design': BlindDesignStudy,
  'prior-guided': PriorGuidedLearning,
  'runtime-cost': RuntimeCost,
  methodology: MethodologyIntegrity,
}

/** URL is the single source of truth for navigation (#story, #dashboard,
 * #dashboard/<tab-id>), so refreshing, bookmarking, sharing a link, or using
 * the browser back/forward buttons all land on the exact same view instead
 * of always resetting to the story page. */
function readRoute() {
  const hash = window.location.hash.replace(/^#\/?/, '')
  const [view, tab] = hash.split('/')
  if (view === 'dashboard') {
    return { view: 'dashboard', tab: tab && TAB_COMPONENTS[tab] ? tab : 'overview' }
  }
  return { view: 'story', tab: 'overview' }
}

/** Fixed ambient glow behind the whole dashboard, in the system's signature
 * accent gradient (violet top-left, fuchsia bottom-right — same two stops as
 * .accent-gradient) rather than one flat radial. Sits above the neutral
 * scroll-gradient-bg (same z-index, later in DOM order so it paints on top)
 * and stays well behind actual content. */
function CategoryGlow() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-[1] overflow-hidden" aria-hidden="true">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: 'easeInOut' }}
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(65% 55% at 12% 0%, rgba(139,92,246,0.24) 0%, transparent 62%), radial-gradient(50% 45% at 100% 100%, rgba(232,121,249,0.16) 0%, transparent 68%)',
        }}
      />
    </div>
  )
}

function DataDashboard({ active, onActiveChange, onBack }) {
  const [paletteOpen, setPaletteOpen] = useState(false)
  const ActiveTab = TAB_COMPONENTS[active]
  const gradientRef = useScrollGradient()

  return (
    <div className="min-h-screen w-full">
      <div ref={gradientRef} className="scroll-gradient-bg" aria-hidden="true" />
      <CategoryGlow />
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} onNavigate={onActiveChange} />
      <DashboardTopBar active={active} onChange={onActiveChange} onBack={onBack} onOpenPalette={() => setPaletteOpen(true)} />
      <div className="mx-auto flex max-w-[1400px] items-start gap-8 px-6">
        <DashboardSidebar active={active} onChange={onActiveChange} />
        <main className="min-w-0 flex-1 py-8 lg:py-10">
          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
            >
              <ActiveTab />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}

function App() {
  const [route, setRoute] = useState(readRoute)

  useEffect(() => {
    const onHashChange = () => setRoute(readRoute())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const goToStory = () => {
    window.location.hash = 'story'
  }
  const goToDashboard = (tab = 'overview') => {
    window.location.hash = `dashboard/${tab}`
  }

  return (
    <MotionConfig reducedMotion="user">
      {route.view === 'dashboard' ? (
        <DataDashboard active={route.tab} onActiveChange={goToDashboard} onBack={goToStory} />
      ) : (
        <StoryPage onOpenDashboard={() => goToDashboard('overview')} />
      )}
    </MotionConfig>
  )
}

export default App
