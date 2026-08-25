import { motion, AnimatePresence } from 'motion/react'
import { CheckCircle2, XCircle } from 'lucide-react'

export function OutcomeBanner({ visible, success }) {
  return (
    <div className="flex h-14 items-center justify-center">
      <AnimatePresence mode="wait">
        {visible && (
          <motion.div
            key={success ? 'success' : 'fail'}
            initial={{ opacity: 0, scale: 0.5, rotate: success ? -8 : 8 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            exit={{ opacity: 0, scale: 0.7 }}
            transition={{ type: 'spring', stiffness: 400, damping: 20 }}
            className={
              'flex items-center gap-2 rounded-full px-5 py-2 font-sans text-base font-semibold ' +
              (success ? 'bg-lav-success/15 text-lav-success' : 'bg-lav-cost/15 text-lav-cost')
            }
          >
            {success ? <CheckCircle2 size={22} /> : <XCircle size={22} />}
            {success ? 'Success' : 'Failed'}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
