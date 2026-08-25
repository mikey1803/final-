import { motion } from 'motion/react'
import {
  LayoutDashboard,
  Target,
  GitCompareArrows,
  Activity,
  Trophy,
  ArrowLeft,
  Zap,
  BookOpen,
  Waves,
  Crosshair,
  EyeOff,
  Gauge,
  ShieldCheck,
  SlidersHorizontal,
  Search,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { LogoBadge } from './Logo'

// Monochrome + one accent: every group used to carry its own brand hue
// (teal / amber / emerald) for its active nav state. Under this system there
// is exactly one accent color anywhere in the UI, so every group key
// resolves to the same lavender — group identity now comes from label, icon,
// and position, not a different active-state color per section. The four
// keys stay (rather than collapsing to one) so every existing
// GROUP_COLORS.<key> / groupColorFor() call site repaints from here alone.
export const GROUP_COLORS = {
  primary: '#8B5CF6',
  teal: '#8B5CF6',
  amber: '#8B5CF6',
  live: '#8B5CF6',
}

export const NAV_GROUPS = [
  {
    label: null,
    color: 'primary',
    tabs: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard, description: 'The headline finding and what every tab below unpacks' },
    ],
  },
  {
    label: 'Core Results',
    color: 'primary',
    tabs: [
      { id: 'baselines', label: 'Baselines Comparison', icon: Trophy, description: 'The full 11-method comparison, adaptive vs. static vs. LLM' },
      { id: 'classifier-noise', label: 'Classifier-Noise Sweep', icon: Target, description: 'How the result holds up as the failure classifier gets noisier' },
      { id: 'non-stationarity', label: 'Non-Stationarity', icon: Activity, description: 'What happens when the environment changes mid-run' },
    ],
  },
  {
    label: 'Generalization & Robustness',
    color: 'teal',
    tabs: [
      { id: 'matrix-family', label: 'Matrix-Family Sensitivity', icon: GitCompareArrows, description: 'Whether the result depends on one hand-picked reward matrix' },
      { id: 'drift-taxonomy', label: 'Drift Taxonomy', icon: Waves, description: 'Every physically-motivated way the environment can drift' },
      { id: 'crossover-predictor', label: 'Crossover Predictor', icon: Crosshair, description: 'Predicting, not just measuring, when adaptive learning wins' },
      { id: 'prior-guided', label: 'Prior-Guided Learning', icon: SlidersHorizontal, description: 'Giving the learner a head start, and when that backfires' },
    ],
  },
  {
    label: 'Validation Studies',
    color: 'amber',
    tabs: [
      { id: 'blind-design', label: 'Blind-Design Study', icon: EyeOff, description: 'Checking the baseline rule against independent AI designs' },
      { id: 'theory-19', label: 'Crossover Theory Validation', icon: BookOpen, description: 'Patching the crossover formula for real-world classifier noise' },
      { id: 'runtime-cost', label: 'Runtime & Cost', icon: Gauge, description: 'Real latency and token spend, not just accuracy' },
      { id: 'methodology', label: 'Methodology & Integrity', icon: ShieldCheck, description: 'Honest audits: a bug found and fixed, a claim corrected' },
    ],
  },
  {
    label: null,
    color: 'live',
    tabs: [{ id: 'live-simulation', label: 'Live Simulation', icon: Zap, description: 'Try the real classifier and selector yourself, live' }],
  },
]

export const TABS = NAV_GROUPS.flatMap((g) => g.tabs)

/** Which group (and therefore which accent color) a given tab id belongs to. */
export function groupColorFor(tabId) {
  const group = NAV_GROUPS.find((g) => g.tabs.some((t) => t.id === tabId))
  return GROUP_COLORS[group?.color ?? 'primary']
}

function Brand() {
  return (
    <div className="flex shrink-0 items-center gap-2.5">
      <LogoBadge size={32} />
      <span className="hidden font-sans text-sm font-semibold tracking-wide text-lav-text sm:inline">
        Failure-Aware Agent
      </span>
    </div>
  )
}

// Active nav state renders as a soft accent-gradient wash (violet -> fuchsia)
// rather than a flat single-color fill — the one accent still reads as one
// identity, just with real depth instead of a flat swatch.
const activeGradientStyle = {
  backgroundImage: 'linear-gradient(135deg, rgba(139,92,246,0.24), rgba(232,121,249,0.16))',
  boxShadow: 'inset 0 0 0 1px rgba(139,92,246,0.45)',
}

function MobileTabPill({ tab, active, onClick }) {
  const Icon = tab.icon
  return (
    <motion.button
      onClick={onClick}
      whileTap={{ scale: 0.94 }}
      transition={tapSpring}
      className={cn(
        'relative flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 font-sans text-xs font-medium transition-colors duration-200',
        active ? 'text-lav-text' : 'text-lav-muted',
      )}
    >
      {active && (
        <motion.div
          layoutId="dashboard-nav-active-pill-mobile"
          className="absolute inset-0 rounded-full"
          style={activeGradientStyle}
          transition={{ type: 'spring', stiffness: 260, damping: 28 }}
        />
      )}
      <Icon size={13} className="relative z-10 shrink-0" />
      <span className="relative z-10 whitespace-nowrap">{tab.label}</span>
    </motion.button>
  )
}

/** Floating "island" top bar — detached from the viewport edge with its own
 * margin, rounded, glass-blurred, shadowed — rather than a flat bar flush
 * against the top. Brand, back-to-story, and (below lg, where the sidebar
 * is hidden) a horizontally scrollable strip of every tab grouped the same
 * way as the sidebar. */
const tapSpring = { type: 'spring', bounce: 0, duration: 0.3 }

export function DashboardTopBar({ active, onChange, onBack, onOpenPalette }) {
  return (
    <header className="sticky top-0 z-20 px-3 pt-3 sm:px-4 sm:pt-4">
      <div className="material-chrome mx-auto max-w-[1400px] rounded-2xl border border-lav-border bg-lav-bg/75 backdrop-blur-xl saturate-150 shadow-[0_12px_36px_rgba(0,0,0,0.5)]">
        <div className="flex items-center justify-between gap-4 px-5 py-2.5">
          <Brand />
          <div className="flex shrink-0 items-center gap-2">
            <motion.button
              onClick={onOpenPalette}
              whileTap={{ scale: 0.96 }}
              transition={tapSpring}
              className="flex items-center gap-2 rounded-full border border-lav-border bg-lav-surface px-3.5 py-1.5 font-sans text-sm text-lav-muted transition-colors hover:border-lav-primary/60 hover:text-lav-text"
            >
              <Search size={14} />
              <span className="hidden sm:inline">Jump to tab</span>
              <kbd className="hidden rounded border border-lav-border px-1.5 py-0.5 font-mono text-[10px] sm:inline">
                {navigator.platform?.includes('Mac') ? '⌘K' : 'Ctrl K'}
              </kbd>
            </motion.button>
            <motion.button
              onClick={onBack}
              whileTap={{ scale: 0.96 }}
              transition={tapSpring}
              className="flex items-center gap-1.5 rounded-full border border-lav-border bg-lav-surface px-3.5 py-1.5 font-sans text-sm text-lav-text transition-colors hover:border-lav-primary/60 hover:bg-lav-primary/10 hover:text-lav-primary"
            >
              <ArrowLeft size={15} />
              <span className="hidden sm:inline">Project Story</span>
            </motion.button>
          </div>
        </div>

        <nav
          className="flex items-center gap-5 overflow-x-auto px-4 pb-2.5 [scrollbar-width:none] lg:hidden [&::-webkit-scrollbar]:hidden"
          style={{ maskImage: 'linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent)' }}
        >
          {NAV_GROUPS.map((group, i) => (
            <div key={group.label ?? `group-${i}`} className="flex shrink-0 items-center gap-1.5">
              {group.label && (
                <span className="mr-0.5 whitespace-nowrap font-sans text-[10px] font-semibold uppercase tracking-wide text-lav-muted/70">
                  {group.label}
                </span>
              )}
              {group.tabs.map((tab) => (
                <MobileTabPill key={tab.id} tab={tab} active={active === tab.id} onClick={() => onChange(tab.id)} />
              ))}
            </div>
          ))}
        </nav>
      </div>
    </header>
  )
}

function SidebarLink({ tab, active, onClick }) {
  const Icon = tab.icon
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ x: active ? 0 : 2 }}
      whileTap={{ scale: 0.98 }}
      transition={tapSpring}
      className="relative flex items-start gap-2.5 rounded-lg px-3 py-2 text-left transition-colors"
    >
      {active && (
        <motion.span
          layoutId="sidebar-active-indicator"
          className="absolute inset-0 rounded-lg"
          style={activeGradientStyle}
          transition={{ type: 'spring', stiffness: 260, damping: 28 }}
        />
      )}
      <motion.span
        animate={active ? { scale: 1.1 } : { scale: 1 }}
        transition={tapSpring}
        className="relative z-10 mt-0.5 shrink-0"
      >
        <Icon size={15} className={active ? 'text-lav-primary' : 'text-lav-muted'} />
      </motion.span>
      <span className="relative z-10 min-w-0">
        <span
          className={cn(
            'block font-sans text-sm',
            active ? 'accent-gradient-text font-semibold' : 'text-lav-muted',
          )}
        >
          {tab.label}
        </span>
        {tab.description && (
          <span className="mt-0.5 block font-sans text-xs leading-snug text-lav-muted/80">{tab.description}</span>
        )}
      </span>
    </motion.button>
  )
}

/** Desktop-only (lg+) grouped vertical nav, sticky below the top bar. The
 * active-state color (fixed above in GROUP_COLORS) is the system's one
 * accent regardless of group; the shared layoutId on the active pill lets
 * Motion morph its position when jumping between groups. */
export function DashboardSidebar({ active, onChange, className }) {
  return (
    <aside className={cn('hidden shrink-0 lg:block lg:w-60', className)}>
      <nav className="sticky top-[88px] flex flex-col gap-5 py-8">
        {NAV_GROUPS.map((group, i) => (
          <div key={group.label ?? `group-${i}`} className="flex flex-col gap-1">
            {group.label && (
              <p className="mb-1 px-3 font-sans text-[11px] font-semibold uppercase tracking-wide text-lav-muted/70">
                {group.label}
              </p>
            )}
            {group.tabs.map((tab) => (
              <SidebarLink key={tab.id} tab={tab} active={active === tab.id} onClick={() => onChange(tab.id)} />
            ))}
          </div>
        ))}
      </nav>
    </aside>
  )
}
