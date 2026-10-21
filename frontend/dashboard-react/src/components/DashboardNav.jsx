import { useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
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
  Swords,
  Stethoscope,
  Grid3x3,
  Printer,
  ChevronDown,
  FlaskConical,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { LogoMark } from './Logo'

// Single navigational accent (--color-lav-primary, an indigo/blue — see
// index.css's design-system note) used for "active/selected" only. Group
// identity comes from label + icon + position, not a rainbow of hues; the
// four color keys stay only so any historical GROUP_COLORS.<key> call site
// still resolves without edits.
export const GROUP_COLORS = {
  primary: '#4C6FFF',
  teal: '#4C6FFF',
  amber: '#4C6FFF',
  live: '#4C6FFF',
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
    label: 'Try It Live',
    color: 'live',
    collapsible: false,
    tabs: [
      { id: 'live-simulation', label: 'Live Simulation', icon: Zap, description: 'Try the real classifier and selector yourself, live' },
      { id: 'beat-the-bandit', label: 'Beat the Bandit', icon: Swords, description: 'Pick recovery strategies blind and race the real bandit, round by round' },
      { id: 'diagnose', label: 'Diagnose an Error', icon: Stethoscope, description: 'Paste any real error text and get a live prediction and recovery recommendation' },
      { id: 'matrix-explorer', label: 'Try Your Own Scenario', icon: FlaskConical, description: 'Edit the hidden success-rate matrix yourself and see if the headline finding still holds' },
    ],
  },
  {
    label: 'Core Results',
    color: 'primary',
    collapsible: true,
    tabs: [
      { id: 'baselines', label: 'Baselines Comparison', icon: Trophy, description: 'The full 11-method comparison, adaptive vs. static vs. LLM' },
      { id: 'classifier-noise', label: 'Classifier-Noise Sweep', icon: Target, description: 'How the result holds up as the failure classifier gets noisier' },
      { id: 'classifier-diagnostics', label: 'Classifier Diagnostics', icon: Grid3x3, description: 'Real confusion matrix and per-class precision/recall/F1 for the trained classifier' },
      { id: 'non-stationarity', label: 'Non-Stationarity', icon: Activity, description: 'What happens when the environment changes mid-run' },
    ],
  },
  {
    label: 'Generalization & Robustness',
    color: 'teal',
    collapsible: true,
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
    collapsible: true,
    tabs: [
      { id: 'blind-design', label: 'Blind-Design Study', icon: EyeOff, description: 'Checking the baseline rule against independent AI designs' },
      { id: 'theory-19', label: 'Crossover Theory Validation', icon: BookOpen, description: 'Patching the crossover formula for real-world classifier noise' },
      { id: 'runtime-cost', label: 'Runtime & Cost', icon: Gauge, description: 'Real latency and token spend, not just accuracy' },
      { id: 'methodology', label: 'Methodology & Integrity', icon: ShieldCheck, description: 'Honest audits: a bug found and fixed, a claim corrected' },
    ],
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
    <div className="flex shrink-0 items-center gap-2">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-lav-border text-lav-text">
        <LogoMark size={13} />
      </span>
      <span className="hidden font-mono text-xs font-medium tracking-tight text-lav-text sm:inline">
        failure-aware-agent
      </span>
    </div>
  )
}

function MobileTabPill({ tab, active, onClick }) {
  const Icon = tab.icon
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative flex shrink-0 items-center gap-1.5 whitespace-nowrap border-b-2 px-0.5 py-2 font-sans text-xs font-medium transition-colors duration-150',
        active ? 'border-lav-primary text-lav-text' : 'border-transparent text-lav-muted hover:text-lav-text',
      )}
    >
      <Icon size={13} className="shrink-0" />
      {tab.label}
    </button>
  )
}

/** Sticky top bar: flush, thin, structural — a strip of instrumentation
 * chrome, not a floating rounded "island". Brand mark, jump-to-tab, and
 * (below lg, where the sidebar is hidden) a horizontally scrollable strip
 * of every tab grouped the same way as the sidebar. */
export function DashboardTopBar({ active, onChange, onBack, onOpenPalette }) {
  return (
    <header className="material-chrome chrome-edge sticky top-0 z-20 border-b border-lav-border bg-lav-bg/92 backdrop-blur-md print:hidden">
      <div className="mx-auto flex h-12 max-w-[1400px] items-center justify-between gap-4 px-5">
        <Brand />
        <div className="flex shrink-0 items-center gap-4">
          <button
            onClick={() => window.print()}
            className="flex items-center gap-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-text"
            title="Print this tab, or save it as a PDF"
          >
            <Printer size={13} />
            <span className="hidden sm:inline">Export</span>
          </button>
          <button
            onClick={onOpenPalette}
            className="flex items-center gap-2 font-sans text-xs text-lav-muted transition-colors hover:text-lav-text"
          >
            <Search size={13} />
            <span className="hidden sm:inline">Jump to tab</span>
            <kbd className="hidden rounded border border-lav-border px-1.5 py-0.5 font-mono text-[10px] text-lav-muted sm:inline">
              {navigator.platform?.includes('Mac') ? '⌘K' : 'Ctrl K'}
            </kbd>
          </button>
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 font-sans text-xs text-lav-muted transition-colors hover:text-lav-text"
          >
            <ArrowLeft size={13} />
            <span className="hidden sm:inline">Project Story</span>
          </button>
        </div>
      </div>

      <nav
        className="flex items-center gap-5 overflow-x-auto border-t border-lav-border px-4 [scrollbar-width:none] lg:hidden [&::-webkit-scrollbar]:hidden"
        style={{ maskImage: 'linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent)' }}
      >
        {NAV_GROUPS.map((group, i) => (
          <div key={group.label ?? `group-${i}`} className="flex shrink-0 items-center gap-3">
            {group.tabs.map((tab) => (
              <MobileTabPill key={tab.id} tab={tab} active={active === tab.id} onClick={() => onChange(tab.id)} />
            ))}
          </div>
        ))}
      </nav>
    </header>
  )
}

function SidebarLink({ tab, active, onClick }) {
  const Icon = tab.icon
  return (
    <button
      onClick={onClick}
      className="group relative flex items-start gap-2.5 py-1.5 pl-3.5 pr-2 text-left"
    >
      <span className="absolute inset-y-0.5 left-0 w-[2px] overflow-hidden rounded-full bg-transparent">
        {active && (
          <motion.span
            layoutId="sidebar-active-indicator"
            className="block h-full w-full bg-lav-primary"
            transition={{ type: 'spring', stiffness: 340, damping: 32 }}
          />
        )}
      </span>
      <Icon size={14} className={cn('mt-0.5 shrink-0 transition-colors', active ? 'text-lav-primary' : 'text-lav-muted group-hover:text-lav-text')} />
      <span className="min-w-0">
        <span className={cn('block font-sans text-sm transition-colors', active ? 'font-medium text-lav-text' : 'text-lav-muted group-hover:text-lav-text')}>
          {tab.label}
        </span>
        {tab.description && (
          <span className="mt-0.5 block font-sans text-xs leading-snug text-lav-muted/70">{tab.description}</span>
        )}
      </span>
    </button>
  )
}

const COLLAPSE_KEY = 'dashboard-nav-collapsed-groups-v1'

function loadCollapsed() {
  try {
    const raw = localStorage.getItem(COLLAPSE_KEY)
    return raw ? new Set(JSON.parse(raw)) : new Set(NAV_GROUPS.filter((g) => g.collapsible).map((g) => g.label))
  } catch {
    return new Set(NAV_GROUPS.filter((g) => g.collapsible).map((g) => g.label))
  }
}

function saveCollapsed(set) {
  try {
    localStorage.setItem(COLLAPSE_KEY, JSON.stringify([...set]))
  } catch {
    // collapse state just won't persist across reloads
  }
}

/** Desktop-only (lg+) grouped vertical nav, sticky below the top bar.
 * Active state is a thin left-edge tick that Motion morphs between items
 * (VS Code / Linear convention), not a filled pill — reserves color for
 * the one thing it needs to say ("you are here"). The three deep research
 * groups (13 tabs combined) collapse to a single header by default so the
 * sidebar opens on ~5 items, not a 17-tab wall — Overview and the
 * interactive "Try It Live" group (the two things worth seeing first)
 * always stay expanded. Whichever group holds the active tab force-opens,
 * so deep-linking or search never lands inside a hidden section. */
export function DashboardSidebar({ active, onChange, className }) {
  const [collapsed, setCollapsed] = useState(loadCollapsed)

  const toggleGroup = (label) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(label)) next.delete(label)
      else next.add(label)
      saveCollapsed(next)
      return next
    })
  }

  return (
    <aside className={cn('hidden shrink-0 print:hidden lg:block lg:w-60', className)}>
      <nav className="sticky top-[49px] flex flex-col gap-5 py-8">
        {NAV_GROUPS.map((group, i) => {
          const isOpen = !group.collapsible || !collapsed.has(group.label) || group.tabs.some((t) => t.id === active)
          return (
            <div key={group.label ?? `group-${i}`} className="flex flex-col gap-0.5">
              {group.label && (
                group.collapsible ? (
                  <button
                    onClick={() => toggleGroup(group.label)}
                    className="mb-1.5 flex items-center justify-between px-3.5 font-mono text-[10px] font-medium uppercase tracking-wider text-lav-muted/60 transition-colors hover:text-lav-muted"
                  >
                    <span>{group.label}</span>
                    <span className="flex items-center gap-1">
                      {!isOpen && <span className="tabular-nums">{group.tabs.length}</span>}
                      <ChevronDown size={11} className={cn('shrink-0 transition-transform duration-200', isOpen && 'rotate-180')} />
                    </span>
                  </button>
                ) : (
                  <p className="mb-1.5 px-3.5 font-mono text-[10px] font-medium uppercase tracking-wider text-lav-muted/60">
                    {group.label}
                  </p>
                )
              )}
              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ type: 'spring', bounce: 0, duration: 0.25 }}
                    className="overflow-hidden"
                  >
                    <div className="flex flex-col gap-0.5">
                      {group.tabs.map((tab) => (
                        <SidebarLink key={tab.id} tab={tab} active={active === tab.id} onClick={() => onChange(tab.id)} />
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
