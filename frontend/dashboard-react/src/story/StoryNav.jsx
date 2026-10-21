import { motion, useScroll, useTransform, useSpring } from 'motion/react'
import { LayoutGrid } from 'lucide-react'
import { cn } from '../lib/utils'
import { useActiveSection } from './useActiveSection'
import { LogoMark } from '../components/Logo'

const LINKS = [
  { id: 'problem', label: 'Problem' },
  { id: 'solution', label: 'Solution' },
  { id: 'demo', label: 'Demo' },
  { id: 'results', label: 'Results' },
]

const tapSpring = { type: 'spring', bounce: 0, duration: 0.25 }

function NavLink({ link, active, onClick, layoutId }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative shrink-0 px-3 py-1.5 font-sans text-sm transition-colors',
        active ? 'text-lav-text' : 'text-lav-muted hover:text-lav-text',
      )}
    >
      {link.label}
      {active && (
        <motion.span
          layoutId={layoutId}
          className="absolute inset-x-2 -bottom-px h-[2px] bg-lav-primary"
          transition={{ type: 'spring', stiffness: 340, damping: 32 }}
        />
      )}
    </button>
  )
}

export function StoryNav({ onOpenDashboard }) {
  const active = useActiveSection(LINKS.map((l) => l.id))

  // Real scroll-driven motion: the bar compresses and its glass gets
  // denser the further you scroll, a continuous response to scroll
  // position rather than a discrete hover/tap trigger, smoothed through a
  // spring so it settles instead of snapping to the raw value.
  const { scrollY } = useScroll()
  const rawH = useTransform(scrollY, [0, 160], [56, 44])
  const barHeight = useSpring(rawH, { stiffness: 300, damping: 40, mass: 0.5 })
  const rawOpacity = useTransform(scrollY, [0, 160], [0.75, 0.94])
  const bgOpacity = useSpring(rawOpacity, { stiffness: 300, damping: 40, mass: 0.5 })
  const bgColor = useTransform(bgOpacity, (v) => `rgba(8,9,10,${v})`)

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <motion.header
      className="material-chrome sticky top-0 z-20 border-b border-lav-border backdrop-blur-md"
      style={{ backgroundColor: bgColor }}
    >
      <motion.div className="mx-auto flex max-w-6xl items-center justify-between px-5" style={{ height: barHeight }}>
        <button
          onClick={() => document.getElementById('hero')?.scrollIntoView({ behavior: 'smooth' })}
          className="flex shrink-0 items-center gap-2"
        >
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-lav-border text-lav-text">
            <LogoMark size={13} />
          </span>
          <span className="font-mono text-xs font-medium tracking-tight text-lav-text">failure-aware-agent</span>
        </button>

        <nav className="hidden items-center gap-1 sm:flex">
          {LINKS.map((link) => (
            <NavLink key={link.id} link={link} active={active === link.id} onClick={() => scrollTo(link.id)} layoutId="story-nav-tick" />
          ))}
        </nav>

        <motion.button
          onClick={onOpenDashboard}
          whileTap={{ scale: 0.96 }}
          transition={tapSpring}
          className="flex shrink-0 items-center gap-1.5 border border-lav-border px-3 py-1.5 font-sans text-xs text-lav-text transition-colors hover:border-lav-primary/60 hover:text-lav-primary"
        >
          <LayoutGrid size={13} />
          <span className="hidden sm:inline">Data Dashboard</span>
        </motion.button>
      </motion.div>

      {/* Mobile: horizontally scrollable link strip, since the centered nav hides below sm. */}
      <nav className="flex items-center gap-1 overflow-x-auto border-t border-lav-border px-4 py-1.5 [scrollbar-width:none] sm:hidden [&::-webkit-scrollbar]:hidden">
        {LINKS.map((link) => (
          <NavLink key={link.id} link={link} active={active === link.id} onClick={() => scrollTo(link.id)} layoutId="story-nav-tick-mobile" />
        ))}
      </nav>
    </motion.header>
  )
}
