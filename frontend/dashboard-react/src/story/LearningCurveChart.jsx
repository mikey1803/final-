import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import { ChartTooltip } from '../components/ChartTooltip'

export function LearningCurveChart({ windows, currentIndex, shiftEpisode, showShift }) {
  const current = windows[currentIndex]

  return (
    <ResponsiveContainer width="100%" height={420}>
      <LineChart data={windows} margin={{ top: 12, right: 24, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
        <XAxis
          dataKey="episodeMid"
          type="number"
          domain={[0, 500]}
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
          stroke="var(--color-lav-border)"
          label={{ value: 'Episode', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--color-lav-muted)' }}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, fill: 'var(--color-lav-muted)' }}
          stroke="var(--color-lav-border)"
        />
        <Tooltip
          content={<ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} labelFormatter={(l) => `~episode ${l}`} />}
          cursor={{ stroke: 'var(--color-lav-primary)', strokeWidth: 1, strokeDasharray: '4 4' }}
        />
        {showShift && (
          <ReferenceLine
            x={shiftEpisode}
            stroke="var(--color-lav-cost)"
            strokeDasharray="4 4"
            label={{ value: 'Disruption', position: 'top', fontSize: 11, fill: 'var(--color-lav-cost)' }}
          />
        )}
        <Line
          type="monotone"
          dataKey="windowSuccessRate"
          name="Success rate"
          stroke="var(--color-lav-primary)"
          strokeWidth={2.5}
          dot={false}
          animationDuration={500} animationEasing="ease-out"
        />
        {current && (
          <ReferenceDot
            x={current.episodeMid}
            y={current.windowSuccessRate}
            r={6}
            fill="var(--color-lav-primary)"
            stroke="var(--color-lav-bg)"
            strokeWidth={2}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
