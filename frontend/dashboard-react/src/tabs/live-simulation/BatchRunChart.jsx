import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { ChartTooltip } from '../../components/ChartTooltip'
import { InteractiveLegend } from '../../components/InteractiveLegend'
import { useSeriesToggle } from '../../hooks/useSeriesToggle'
import { colorFor } from '../../lib/colors'

/** Pass `hasShiftedSeries` once the batch-run response actually carries
 * `*_shifted` fields with at least one non-null value (shift was active and
 * at least one shifted-type episode landed) — see api/main.py's batch_run.
 * Shifted-type-only lines render bold/solid (the honest recovery signal,
 * same reasoning as the Non-Stationarity tab's shifted-type-only chart);
 * all-types lines render thin/dashed alongside them for context, both
 * toggleable via the shared interactive legend. */
export function BatchRunChart({ rows, hasShiftedSeries = false }) {
  const series = useSeriesToggle()
  const dim = (key) => (series.hoveredKey && series.hoveredKey !== key ? 0.25 : 1)

  return (
    <ResponsiveContainer width="100%" height={hasShiftedSeries ? 260 : 220}>
      <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--color-lav-border)" vertical={false} />
        <XAxis
          dataKey="episode"
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: 'var(--color-lav-muted)' }}
          stroke="var(--color-lav-border)"
        />
        <YAxis
          domain={[0, 1]}
          allowDataOverflow
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: 'var(--color-lav-muted)' }}
          stroke="var(--color-lav-border)"
          width={40}
        />
        <Tooltip
          content={<ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} labelFormatter={(l) => `episode ${l}`} />}
          cursor={{ stroke: 'var(--color-lav-primary)', strokeWidth: 1, strokeDasharray: '4 4' }}
        />
        <Legend
          content={<InteractiveLegend hidden={series.hidden} onToggle={series.toggle} onHover={series.setHovered} />}
        />
        {hasShiftedSeries && (
          <Line
            type="monotone"
            dataKey="thompson_success_rate_shifted"
            name="Thompson Sampling (shifted types)"
            stroke={colorFor('Adaptive (Thompson)')}
            strokeWidth={2.8}
            strokeOpacity={dim('thompson_success_rate_shifted')}
            dot={false}
            hide={series.isHidden('thompson_success_rate_shifted')}
            animationDuration={500} animationEasing="ease-out"
            connectNulls
          />
        )}
        {hasShiftedSeries && (
          <Line
            type="monotone"
            dataKey="fixed_rule_success_rate_shifted"
            name="Fixed Rule (shifted types)"
            stroke={colorFor('Fixed Rule')}
            strokeWidth={2.8}
            strokeOpacity={dim('fixed_rule_success_rate_shifted')}
            dot={false}
            hide={series.isHidden('fixed_rule_success_rate_shifted')}
            animationDuration={500} animationEasing="ease-out"
            connectNulls
          />
        )}
        <Line
          type="monotone"
          dataKey="thompson_success_rate"
          name={hasShiftedSeries ? 'Thompson Sampling (all types)' : 'Thompson Sampling'}
          stroke={colorFor('Adaptive (Thompson)')}
          strokeWidth={hasShiftedSeries ? 1.5 : 2.5}
          strokeDasharray={hasShiftedSeries ? '3 3' : undefined}
          strokeOpacity={hasShiftedSeries ? 0.55 * dim('thompson_success_rate') : dim('thompson_success_rate')}
          dot={false}
          hide={series.isHidden('thompson_success_rate')}
          animationDuration={500} animationEasing="ease-out"
        />
        <Line
          type="monotone"
          dataKey="fixed_rule_success_rate"
          name={hasShiftedSeries ? 'Fixed Rule (all types)' : 'Fixed Rule'}
          stroke={colorFor('Fixed Rule')}
          strokeWidth={hasShiftedSeries ? 1.5 : 2}
          strokeDasharray={hasShiftedSeries ? '3 3' : '4 4'}
          strokeOpacity={hasShiftedSeries ? 0.55 * dim('fixed_rule_success_rate') : dim('fixed_rule_success_rate')}
          dot={false}
          hide={series.isHidden('fixed_rule_success_rate')}
          animationDuration={500} animationEasing="ease-out"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
