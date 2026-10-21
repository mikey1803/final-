import { ChevronRight } from 'lucide-react'

/** The natural-language request field, styled as a real task submission
 * (a prompt glyph + monospace) instead of a generic labeled form field —
 * this is the event that enters the system, not paperwork. */
export function TaskInput({ value, onChange, onSubmit, disabled }) {
  return (
    <div className="flex items-center gap-2 rounded-md border border-lav-border bg-lav-bg px-3 py-2 focus-within:border-lav-primary/50">
      <ChevronRight size={14} className="shrink-0 text-lav-muted" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && !disabled && onSubmit()}
        placeholder="Describe the task you want the agent to perform..."
        disabled={disabled}
        className="w-full bg-transparent font-mono text-sm text-lav-text placeholder:text-lav-muted/60 focus:outline-none disabled:opacity-50"
      />
    </div>
  )
}
