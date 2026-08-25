import { motion, useScroll, useTransform, useSpring } from 'motion/react'
import { LayoutGrid } from 'lucide-react'
import { cn } from '../lib/utils'
import { useActiveSection } from './useActiveSection'
import { LogoBadge } from '../components/Logo'

const LINKS = [
  { id: 'problem', label: 'Problem' },
  { id: 'solution', label: 'Solution' },
  { id: 'demo', label: 'Demo' },
  { id: 'results', label: 'Results' },
]

const tapSpring = { type: 'spring', bounce: 0, duration: 0.3 }

const activeGradientStyle = {
  backgroundImage: 'linear-gradient(135deg, rgba(139,92,246,0.22), rgba(232,121,249,0.16))',
  boxShadow: 'inset 0 0 0 1px rgba(139,92,246,0.4)',
}

function NavLink({ link, active, onClick, className, layoutId }) {
  return (
    <motion.button
      onClick={onClick}
      whileTap={{ scale: 0.95 }}
      transition={tapSpring}
      className={cn(
        'relative shrink-0 rounded-full px-3.5 py-1.5 font-sans text-sm transition-colors',
        active ? 'text-lav-text' : 'text-lav-muted hover:text-lav-text',
        className,
      )}
    >
      {active && (
        <motion.span
          layoutId={layoutId}
          className="absolute inset-0 rounded-full"
          style={activeGradientStyle}
          transition={{ type: 'spring', stiffness: 260, damping: 28 }}
        />
      )}
      <span className="relative z-10">{link.label}</span>
    </motion.button>
  )
}

export function StoryNav({ onOpenDashboard }) {
  const active = useActiveSection(LINKS.map((l) => l.id))

  // Real scroll-driven motion, not just discrete hover/tap springs: the
  // floating pill compresses and its glass gets denser the further you
  // scroll, the same "fluid" continuous-response feel as the rest of the
  // interface, smoothed through a spring so it settles rather than
  // snapping to the raw scroll position.
  const { scrollY } = useScroll()
  const rawPadY = useTransform(scrollY, [0, 160], [14, 7])
  const paddingY = useSpring(rawPadY, { stiffness: 300, damping: 40, mass: 0.5 })
  const rawOpacity = useTransform(scrollY, [0, 160], [0.75, 0.94])
  const bgOpacity = useSpring(rawOpacity, { stiffness: 300, damping: 40, mass: 0.5 })
  const bgColor = useTransform(bgOpacity, (v) => `rgba(9,9,11,${v})`)

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <header className="sticky top-0 z-20 px-3 pt-3 sm:px-4 sm:pt-4">
      <motion.div
        className="material-chrome mx-auto max-w-6xl rounded-2xl border border-lav-border backdrop-blur-xl saturate-150 shadow-[0_12px_36px_rgba(0,0,0,0.5)]"
        style={{ backgroundColor: bgColor }}
      >
        <motion.div
          className="flex items-center justify-between px-5"
          style={{ paddingTop: paddingY, paddingBottom: paddingY }}
        >
          <motion.button
            onClick={() => document.getElementById('hero')?.scrollIntoView({ behavior: 'smooth' })}
            whileTap={{ scale: 0.96 }}
            transition={tapSpring}
            className="flex shrink-0 items-center gap-2"
          >
            <LogoBadge size={28} />
            <span className="font-sans text-sm font-semibold text-lav-text">Failure-Aware Agent</span>
          </motion.button>

          <nav className="hidden items-center gap-1 sm:flex">
            {LINKS.map((link) => (
              <NavLink
                key={link.id}
                link={link}
                active={active === link.id}
                onClick={() => scrollTo(link.id)}
                layoutId="story-nav-active-pill-desktop"
              />
            ))}
          </nav>

          <motion.button
            onClick={onOpenDashboard}
            whileTap={{ scale: 0.96 }}
            transition={tapSpring}
            className="accent-gradient flex shrink-0 items-center gap-1.5 rounded-full px-3.5 py-1.5 font-sans text-sm font-medium text-lav-bg"
          >
            <LayoutGrid size={15} />
            <span className="hidden sm:inline">Data Dashboard</span>
          </motion.button>
        </motion.div>

        {/* Mobile: horizontally scrollable link strip, since the centered nav hides below sm. */}
        <nav className="flex items-center gap-1 overflow-x-auto px-4 pb-2.5 [scrollbar-width:none] sm:hidden [&::-webkit-scrollbar]:hidden">
          {LINKS.map((link) => (
            <NavLink
              key={link.id}
              link={link}
              active={active === link.id}
              onClick={() => scrollTo(link.id)}
              className="px-3 py-1 text-xs"
              layoutId="story-nav-active-pill-mobile"
            />
          ))}
        </nav>
      </motion.div>
    </header>
  )
}
