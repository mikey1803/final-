import { motion } from 'motion/react'

/** Vertical connector line drawn between two pipeline layers as the lower one mounts. */
export function PipelineConnector() {
  return (
    <motion.div
      initial={{ opacity: 0, scaleY: 0 }}
      animate={{ opacity: 1, scaleY: 1 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="mx-auto h-6 w-px origin-top bg-lav-primary/30"
    />
  )
}

/** One bordered stage of the recovery pipeline, with a pulsing border/dot while `processing`. */
export function PipelineLayer({ number, name, processing, children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{
        opacity: 1,
        y: 0,
        boxShadow: processing
          ? ['0 0 0px rgba(167,139,250,0)', '0 0 14px rgba(167,139,250,0.22)', '0 0 0px rgba(167,139,250,0)']
          : '0 0 0px rgba(167,139,250,0)',
      }}
      transition={
        processing
          ? { boxShadow: { duration: 1.1, repeat: Infinity, ease: 'easeInOut' }, opacity: { duration: 0.3 }, y: { duration: 0.3 } }
          : { duration: 0.3, ease: 'easeOut' }
      }
      className={
        'rounded-xl border-2 px-4 py-3.5 transition-colors duration-300 ' +
        (processing ? 'border-lav-primary/50 bg-lav-primary/[0.04]' : 'border-lav-border bg-lav-surface')
      }
    >
      <div className="mb-2.5 flex items-center gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-lav-primary/15 font-mono text-[10px] font-bold text-lav-primary">
          {number}
        </span>
        <p className="font-sans text-xs font-semibold uppercase tracking-wide text-lav-muted">
          Layer {number}: {name}
        </p>
        {processing && (
          <span className="ml-auto flex items-center gap-1.5 font-mono text-[11px] text-lav-primary">
            <motion.span
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
              className="h-1.5 w-1.5 rounded-full bg-lav-primary"
            />
            Processing...
          </span>
        )}
      </div>
      {children}
    </motion.div>
  )
}
