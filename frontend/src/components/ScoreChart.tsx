import { useEffect, useRef, useState } from 'react'
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

const SQUEEZE_MS = 700
const MAX_TICKS = 10

type Props = {
  iterations: JobIteration[]
  higherIsBetter: boolean
  scoreLabel: string
  selectedIteration: number | null
  onSelect: (iteration: number) => void
}

type ChartPoint = { iteration: number; score: number | null; best: number | null }

function bestSoFarSeries(
  iterations: JobIteration[],
  higherIsBetter: boolean,
): ChartPoint[] {
  let best: number | null = null
  return iterations.map((step) => {
    if (step.score == null) {
      return { iteration: step.iteration, score: null, best: null }
    }
    if (best == null) {
      best = step.score
    } else if (higherIsBetter) {
      best = Math.max(best, step.score)
    } else {
      best = Math.min(best, step.score)
    }
    return { iteration: step.iteration, score: step.score, best }
  })
}

type Range = { min: number; max: number }

/**
 * Eases each bound toward its target. Values past the current bounds stay
 * clipped, so a range that grows reveals the new extreme from that edge
 * while the existing curve compresses with the axis.
 */
function useEasedRange(target: Range): Range {
  const [range, setRange] = useState(target)
  const rangeRef = useRef(target)
  const { min: targetMin, max: targetMax } = target

  useEffect(() => {
    const from = rangeRef.current
    if (from.min === targetMin && from.max === targetMax) {
      return
    }
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const start = performance.now()
    let frame = 0
    const tick = (now: number) => {
      const t = reduceMotion ? 1 : Math.min((now - start) / SQUEEZE_MS, 1)
      const eased = 1 - (1 - t) ** 3
      const next =
        t === 1
          ? { min: targetMin, max: targetMax }
          : {
              min: from.min + (targetMin - from.min) * eased,
              max: from.max + (targetMax - from.max) * eased,
            }
      rangeRef.current = next
      setRange(next)
      if (t < 1) {
        frame = requestAnimationFrame(tick)
      }
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [targetMin, targetMax])

  return range
}

function useSqueezedEdge(target: number): number {
  return useEasedRange({ min: 0, max: target }).max
}

/** Pad the data extrema so the curve does not sit on the frame. */
function scoreDomain(data: ChartPoint[]): Range {
  let min = Infinity
  let max = -Infinity
  for (const point of data) {
    for (const value of [point.score, point.best]) {
      if (value == null) {
        continue
      }
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 }
  }
  const span = max - min
  const pad = span === 0 ? Math.max(Math.abs(max) * 0.05, 1e-3) : span * 0.08
  return { min: min - pad, max: max + pad }
}

/** Stable tick values for the settled domain, limited to what is visible mid-ease. */
function ticksInRange(target: Range, visible: Range): number[] {
  const span = Math.max(target.max - target.min, 1e-9)
  const rough = span / 4
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const fraction = rough / magnitude
  const nice = fraction >= 5 ? 10 : fraction >= 2 ? 5 : fraction >= 1 ? 2 : 1
  const step = nice * magnitude
  const ticks: number[] = []
  const start = Math.ceil((target.min - step * 1e-9) / step) * step
  for (let value = start; value <= target.max + step * 1e-6; value += step) {
    if (value < visible.min - step * 1e-6 || value > visible.max + step * 1e-6) {
      continue
    }
    ticks.push(Number(value.toPrecision(12)))
  }
  return ticks
}

function integerTicks(first: number, edge: number, last: number): number[] {
  const span = Math.max(last - first, 1)
  const step = [1, 2, 5, 10, 20, 50].find((s) => span / s <= MAX_TICKS) ?? 100
  const ticks: number[] = []
  for (let i = first; i <= edge + 1e-9; i += 1) {
    if (i === first || i % step === 0) {
      ticks.push(i)
    }
  }
  return ticks
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
  const first = data[0]?.iteration ?? 1
  const last = data[data.length - 1]?.iteration ?? 1
  const edge = Math.max(useSqueezedEdge(last), first + 1)
  const yTarget = scoreDomain(data)
  const yRange = useEasedRange(yTarget)

  if (data.length === 0) {
    return <p className="muted">No steps yet.</p>
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
            type="number"
            domain={[first, edge]}
            allowDataOverflow
            ticks={integerTicks(first, edge, Math.max(last, first + 1))}
            label={{ value: 'Iteration', position: 'insideBottom', offset: -2 }}
          />
          <YAxis
            type="number"
            domain={[yRange.min, yRange.max]}
            allowDataOverflow
            ticks={ticksInRange(yTarget, yRange)}
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
            connectNulls
            isAnimationActive={false}
            dot={(props) => {
              const { cx, cy, payload } = props
              if (cx == null || cy == null || payload.score == null) {
                return <g key={payload.iteration} />
              }
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
            connectNulls
            isAnimationActive={false}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
