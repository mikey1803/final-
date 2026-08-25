/** Shared Recharts tooltip styled to match the lavender design system. */
export function ChartTooltip({ active, payload, label, formatter, labelFormatter }) {
  if (!active || !payload?.length) return null
  // Hide range/band series (e.g. CI shading, dataKey values that are arrays) from the tooltip list.
  const visible = payload.filter((p) => !Array.isArray(p.value))
  if (!visible.length) return null
  return (
    <div className="rounded-lg border border-lav-border bg-lav-surface px-3 py-2 shadow-lg">
      {label !== undefined && (
        <p className="mb-1 font-mono text-xs text-lav-muted">
          {labelFormatter ? labelFormatter(label) : label}
        </p>
      )}
      {visible.map((p) => (
        <div key={p.dataKey ?? p.name} className="flex items-center gap-2 font-sans text-xs">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-lav-text">{p.name}:</span>
          <span className="font-mono font-semibold text-lav-text">
            {formatter ? formatter(p.value) : p.value}
          </span>
        </div>
      ))}
    </div>
  )
}
