import { useMemo, useState } from 'react'

/**
 * Backs a multi-series chart's interactive legend: which series are hidden
 * (click to toggle) and which one is currently hovered (dims the rest).
 * One hook, reused by every chart with an <InteractiveLegend/> instead of
 * duplicating this useState pair per chart.
 */
export function useSeriesToggle() {
  const [hidden, setHidden] = useState(() => new Set())
  const [hoveredKey, setHoveredKey] = useState(null)

  return useMemo(
    () => ({
      hidden,
      hoveredKey,
      isHidden: (key) => hidden.has(key),
      toggle: (key) =>
        setHidden((prev) => {
          const next = new Set(prev)
          if (next.has(key)) next.delete(key)
          else next.add(key)
          return next
        }),
      setHovered: setHoveredKey,
    }),
    [hidden, hoveredKey],
  )
}
