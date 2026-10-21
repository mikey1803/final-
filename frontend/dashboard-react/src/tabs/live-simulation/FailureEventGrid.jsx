import { cn } from '../../lib/utils'
import { FAILURE_TYPES } from '../../data/failureTypes'

/** Failure-type picker as an event grid, not a <select>. Each cell is a
 * real system event: icon (its own fixed identity color, consistent with
 * every chart elsewhere that plots this failure type), label, and a
 * one-line implication. Selection reads as a left-edge tick in the type's
 * own color plus a background lift — no borrowed brand hue. */
export function FailureEventGrid({ selected, onSelect }) {
  return (
    <div className="grid grid-cols-1 gap-px overflow-hidden rounded-md border border-lav-border bg-lav-border sm:grid-cols-2">
      {FAILURE_TYPES.map((f) => {
        const active = selected === f.id
        return (
          <button
            key={f.id}
            onClick={() => onSelect(f.id)}
            className={cn(
              'relative flex items-start gap-2.5 bg-lav-surface px-3 py-2.5 text-left transition-colors',
              active ? 'bg-lav-surface-raised' : 'hover:bg-lav-surface-raised/60',
            )}
          >
            <span
              className="absolute inset-y-0 left-0 w-[2px]"
              style={{ backgroundColor: active ? f.color : 'transparent' }}
            />
            <f.icon size={15} className="mt-0.5 shrink-0" style={{ color: active ? f.color : 'var(--color-lav-muted)' }} />
            <span className="min-w-0">
              <span className={cn('block font-sans text-xs font-medium', active ? 'text-lav-text' : 'text-lav-muted')}>
                {f.label}
              </span>
              <span className="mt-0.5 block font-sans text-[11px] leading-snug text-lav-muted/70">
                {f.description}
              </span>
            </span>
          </button>
        )
      })}
    </div>
  )
}
