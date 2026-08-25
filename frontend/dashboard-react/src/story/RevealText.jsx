import { useRef } from 'react'
import { useInView } from 'motion/react'
import { TextEffect } from '../components/motion-primitives/TextEffect'

/** Headline wrapper: word-by-word TextEffect reveal, fired once when scrolled
 * into view (not autoplay). Used sparingly — section headlines only.
 *
 * Uses `amount` (fraction of the element visible) rather than a shrunk
 * `margin`, which is more permissive: with `once: true`, an element that
 * happens to land near the bottom edge of the viewport on first mount (e.g.
 * this component remounting after the Data Dashboard round-trip, at scrollY
 * 0) must still count as "in view" on that very first check, or it never
 * mounts at all. A negative margin can shrink the detection band past a
 * heading sitting close to the edge and permanently strand it hidden. */
export function RevealText({ children, as = 'h2', className, per = 'word', preset = 'fade-in-blur' }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.1 })

  return (
    <div ref={ref}>
      <TextEffect as={as} per={per} preset={preset} trigger={inView} className={className} speedReveal={1.4}>
        {children}
      </TextEffect>
    </div>
  )
}
