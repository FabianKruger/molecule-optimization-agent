import { structureUrl } from '../api'
import { formatScore } from '../format'
import type { JobIteration } from '../types'

type Props = {
  jobId: string
  iteration: JobIteration | null
  title: string
  emptyMessage: string
  showReason?: boolean
  showScores?: boolean
  showExplanation?: boolean
}

export function MoleculePanel({
  jobId,
  iteration,
  title,
  emptyMessage,
  showReason = false,
  showScores = false,
  showExplanation = false,
}: Props) {
  if (!iteration) {
    return (
      <section className="panel">
        <h2>{title}</h2>
        <p className="muted">{emptyMessage}</p>
      </section>
    )
  }

  const explanation = iteration.explanation?.trim() ?? ''
  const scores = iteration.scores

  return (
    <section className="panel">
      <h2>{title}</h2>
      <div className="molecule">
        <img
          src={structureUrl(jobId, iteration.iteration)}
          alt={`Structure for iteration ${iteration.iteration}`}
          className="structure"
        />
        <div className="molecule-meta">
          <p>
            <strong>Iteration</strong> {iteration.iteration}
          </p>
          <p>
            <strong>Score</strong> {formatScore(iteration.score)}
          </p>
          <p className="smiles">
            <strong>SMILES</strong> {iteration.smiles}
          </p>
          {showScores && scores && Object.keys(scores).length > 0 ? (
            <ul className="score-list">
              {Object.entries(scores).map(([name, value]) => (
                <li key={name}>
                  {name}: {formatScore(value)}
                </li>
              ))}
            </ul>
          ) : null}
          {showReason ? (
            <p>
              <strong>Reason</strong> {iteration.reason || '—'}
            </p>
          ) : null}
          {showExplanation && explanation ? (
            <details>
              <summary>Oracle explanation</summary>
              <pre className="explanation">{explanation}</pre>
            </details>
          ) : null}
        </div>
      </div>
    </section>
  )
}
