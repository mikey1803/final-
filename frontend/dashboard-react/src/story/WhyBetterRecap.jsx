import { Lock, TrendingUp } from 'lucide-react'
import { Card } from '../components/ui/Card'

export function WhyBetterRecap() {
  return (
    <Card hover={false} className="mx-auto max-w-2xl divide-y divide-lav-border">
      <div className="flex items-center gap-3 py-3 first:pt-0">
        <Lock size={16} className="shrink-0 text-lav-cost" />
        <p className="font-sans text-sm text-lav-text">
          <span className="font-medium">Static rule:</span>{' '}
          <span className="text-lav-muted">same output every time: rigid, never reconsiders.</span>
        </p>
      </div>
      <div className="flex items-center gap-3 py-3 last:pb-0">
        <TrendingUp size={16} className="shrink-0 text-lav-success" />
        <p className="font-sans text-sm text-lav-text">
          <span className="font-medium">Online learning:</span>{' '}
          <span className="text-lav-muted">changes over time: adapts as evidence comes in.</span>
        </p>
      </div>
    </Card>
  )
}
