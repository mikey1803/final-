import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Search } from 'lucide-react'
import { NAV_GROUPS, GROUP_COLORS } from './DashboardNav'

const ALL_TABS = NAV_GROUPS.flatMap((g) => g.tabs.map((t) => ({ ...t, groupColor: GROUP_COLORS[g.color] })))

/** Global Cmd/Ctrl+K jump-to-tab palette. `open`/`onOpenChange` are
 * controlled from the parent so a visible top-bar button and the keyboard
 * shortcut can both drive the same state. */
export function CommandPalette({ open, onOpenChange, onNavigate }) {
  const [query, setQuery] = useState('')
  const [highlighted, setHighlighted] = useState(0)
  const inputRef = useRef(null)

  useEffect(() => {
    function onKeyDown(e) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        onOpenChange((o) => !o)
      } else if (e.key === 'Escape') {
        onOpenChange(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onOpenChange])

  useEffect(() => {
    if (open) {
      setQuery('')
      setHighlighted(0)
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  const filtered = query
    ? ALL_TABS.filter((t) => t.label.toLowerCase().includes(query.toLowerCase()))
    : ALL_TABS

  const select = (id) => {
    onNavigate(id)
    onOpenChange(false)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlighted((i) => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlighted((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && filtered[highlighted]) {
      select(filtered[highlighted].id)
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[15vh] backdrop-blur-sm"
          onClick={() => onOpenChange(false)}
        >
          <motion.div
            initial={{ opacity: 0, y: -12, scale: 0.96, filter: 'blur(8px)' }}
            animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
            exit={{ opacity: 0, y: -8, scale: 0.97, filter: 'blur(6px)' }}
            transition={{ type: 'spring', bounce: 0, duration: 0.35 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-lg overflow-hidden rounded-xl border border-lav-border bg-lav-surface/90 shadow-[0_20px_60px_rgba(0,0,0,0.5)] backdrop-blur-xl"
          >
            <div className="flex items-center gap-2.5 border-b border-lav-border px-4 py-3">
              <Search size={16} className="shrink-0 text-lav-muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value)
                  setHighlighted(0)
                }}
                onKeyDown={handleKeyDown}
                placeholder="Jump to a tab..."
                className="w-full bg-transparent font-sans text-sm text-lav-text placeholder:text-lav-muted focus:outline-none"
              />
              <kbd className="shrink-0 rounded border border-lav-border px-1.5 py-0.5 font-mono text-[10px] text-lav-muted">
                Esc
              </kbd>
            </div>
            <div className="max-h-80 overflow-y-auto p-1.5">
              {filtered.length === 0 && (
                <p className="px-3 py-6 text-center font-sans text-sm text-lav-muted">No matching tab.</p>
              )}
              {filtered.map((t, i) => {
                const Icon = t.icon
                return (
                  <motion.button
                    key={t.id}
                    onClick={() => select(t.id)}
                    onMouseEnter={() => setHighlighted(i)}
                    whileTap={{ scale: 0.98 }}
                    transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
                    className="relative flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left font-sans text-sm text-lav-text"
                  >
                    {i === highlighted && (
                      <motion.div
                        layoutId="command-palette-highlight"
                        className="absolute inset-0 rounded-lg bg-lav-surface-raised"
                        transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                      />
                    )}
                    <Icon size={15} className="relative z-10 shrink-0" style={{ color: t.groupColor }} />
                    <span className="relative z-10">{t.label}</span>
                  </motion.button>
                )
              })}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
