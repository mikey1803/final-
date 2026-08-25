import { useEffect, useRef } from 'react'

/**
 * Drives --scroll-gradient-pos (0%-100%) on a ref'd element from window scroll
 * position, rAF-throttled so it never fires more than once per frame.
 *
 * No-ops entirely under prefers-reduced-motion — the element keeps whatever
 * static background-position is set in CSS.
 */
export function useScrollGradient() {
  const ref = useRef(null)

  useEffect(() => {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduceMotion || !ref.current) return undefined

    let ticking = false
    const update = () => {
      ticking = false
      const el = ref.current
      if (!el) return
      const max = document.documentElement.scrollHeight - window.innerHeight
      const progress = max > 0 ? window.scrollY / max : 0
      const clamped = Math.min(100, Math.max(0, progress * 100))
      el.style.setProperty('--scroll-gradient-pos', `${clamped}%`)
    }
    const onScroll = () => {
      if (!ticking) {
        ticking = true
        requestAnimationFrame(update)
      }
    }

    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [])

  return ref
}
