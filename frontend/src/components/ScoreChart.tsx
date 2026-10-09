import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { JobIteration } from '../types'

type Props = {
  iterations: JobIteration[]
  higherIsBetter: boolean
  scoreLabel: string
  selectedIteration: number | null
  onSelect: (iteration: number) => void
}

function bestSoFarSeries(
  iterations: JobIteration[],
  higherIsBetter: boolean,
): Array<{ iteration: number; score: number; best: number }> {
  let best: number | null = null
  return iterations.map((step) => {
    if (best == null) {
      best = step.score
    } else if (higherIsBetter) {
      best = Math.max(best, step.score)
    } else {
      best = Math.min(best, step.score)
    }
    return {
      iteration: step.iteration,
      score: step.score,
      best,
    }
  })
}

export function ScoreChart({
  iterations,
  higherIsBetter,
  scoreLabel,
  selectedIteration,
  onSelect,
}: Props) {
  const data = bestSoFarSeries(iterations, higherIsBetter)
  const direction = higherIsBetter ? 'higher is better' : 'lower is better'

  if (data.length === 0) {
    return <p className="muted">No scored steps yet.</p>
  }

  return (
    <div className="chart">
      <p className="chart-label">
        {scoreLabel} ({direction})
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart
          data={data}
          margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
          onClick={(state) => {
            const payload = (
              state as { activePayload?: Array<{ payload?: { iteration?: number } }> }
            ).activePayload?.[0]?.payload
            const iteration = payload?.iteration
            if (typeof iteration === 'number') {
              onSelect(iteration)
            }
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#d7dce3" />
          <XAxis
            dataKey="iteration"
            allowDecimals={false}
            label={{ value: 'Iteration', position: 'insideBottom', offset: -2 }}
          />
          <YAxis
            label={{
              value: scoreLabel,
              angle: -90,
              position: 'insideLeft',
              style: { textAnchor: 'middle' },
            }}
          />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="score"
            name="Score"
            stroke="#1f6f5b"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload } = props
              const active = payload.iteration === selectedIteration
              return (
                <circle
                  key={payload.iteration}
                  cx={cx}
                  cy={cy}
                  r={active ? 6 : 4}
                  fill={active ? '#0f3d32' : '#1f6f5b'}
                  stroke="#fff"
                  strokeWidth={1}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelect(payload.iteration)}
                />
              )
            }}
            activeDot={{ r: 7 }}
          />
          <Line
            type="monotone"
            dataKey="best"
            name="Best so far"
            stroke="#b45309"
            strokeWidth={2}
            strokeDasharray="6 4"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
