import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { cancelJob, fetchJob } from '../api'
import { FirstMessageDisclosure } from '../components/FirstMessageDisclosure'
import { MoleculePanel } from '../components/MoleculePanel'
import { ScoreChart } from '../components/ScoreChart'
import { formatScore, shortenSmiles } from '../format'
import type { JobDetail, JobIteration } from '../types'

function findIteration(
  iterations: JobIteration[],
  iteration: number | null,
): JobIteration | null {
  if (iteration == null) {
    return null
  }
  return iterations.find((step) => step.iteration === iteration) ?? null
}

function isActiveStatus(status: JobDetail['status']): boolean {
  return status === 'queued' || status === 'running'
}

export function JobPage() {
  const { id = '' } = useParams()
  const [job, setJob] = useState<JobDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [selectedIteration, setSelectedIteration] = useState<number | null>(null)

  async function onCancel() {
    if (!id || cancelling) {
      return
    }
    setCancelling(true)
    setCancelError(null)
    try {
      await cancelJob(id)
    } catch (err) {
      setCancelError(err instanceof Error ? err.message : 'Failed to cancel job')
      setCancelling(false)
    }
  }

  useEffect(() => {
    if (!id) {
      return
    }

    let cancelled = false
    let timer: number | undefined

    async function load() {
      try {
        const detail = await fetchJob(id)
        if (cancelled) {
          return
        }
        setJob(detail)
        setError(null)
        if (!isActiveStatus(detail.status)) {
          setCancelling(false)
        }
        setSelectedIteration((current) => {
          if (current != null) {
            const stillPresent = detail.iterations.some(
              (step) => step.iteration === current,
            )
            if (stillPresent) {
              return current
            }
          }
          return detail.best?.iteration ?? null
        })

        if (isActiveStatus(detail.status)) {
          timer = window.setTimeout(() => {
            void load()
          }, 2000)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load job')
        }
      }
    }

    void load()
    return () => {
      cancelled = true
      if (timer != null) {
        window.clearTimeout(timer)
      }
    }
  }, [id])

  const selected = useMemo(
    () => (job ? findIteration(job.iterations, selectedIteration) : null),
    [job, selectedIteration],
  )

  const bestStep = useMemo(() => {
    if (!job?.best) {
      return null
    }
    return findIteration(job.iterations, job.best.iteration)
  }, [job])

  if (error) {
    return (
      <div className="page">
        <p className="error" role="alert">
          {error}
        </p>
        <Link to="/">Back to launch</Link>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="page">
        <p className="muted">Loading job…</p>
      </div>
    )
  }

  const statusLine =
    job.status === 'queued'
      ? job.queue_position != null
        ? `waiting · #${job.queue_position}`
        : 'waiting'
      : job.status === 'running'
        ? `${job.phase ?? 'running'} · ${job.iterations_completed}/${job.max_iterations}`
        : job.status

  const showTrajectory = job.status !== 'queued'

  return (
    <div className="page">
      <header className="page-header">
        <p className="crumb">
          <Link to="/">← Launch</Link>
        </p>
        <h1>{job.preset_title}</h1>
        <p>
          <span className={`status status-${job.status}`}>{statusLine}</span>
        </p>
        {job.status === 'queued' ? (
          <p className="muted">
            This job is waiting for the current run to finish
            {job.queue_position != null
              ? ` (position ${job.queue_position} in the queue)`
              : ''}
            .
          </p>
        ) : null}
        {isActiveStatus(job.status) ? (
          <p>
            <button type="button" onClick={() => void onCancel()} disabled={cancelling}>
              {cancelling
                ? 'Cancelling…'
                : job.status === 'queued'
                  ? 'Cancel queued job'
                  : 'Cancel run'}
            </button>
          </p>
        ) : null}
        {cancelError ? (
          <p className="error" role="alert">
            {cancelError}
          </p>
        ) : null}
        {job.status === 'cancelled' ? (
          <p className="muted">This run was cancelled.</p>
        ) : null}
        {job.status === 'failed' && job.error ? (
          <p className="error" role="alert">
            {job.error}
          </p>
        ) : null}
      </header>

      <section className="panel">
        <FirstMessageDisclosure message={job.first_message} />
      </section>

      {showTrajectory ? (
        <>
          <MoleculePanel
            jobId={job.id}
            iteration={bestStep}
            title="Best so far"
            emptyMessage="No molecule has been scored yet."
          />

          <section className="panel">
            <h2>Trajectory</h2>
            <ScoreChart
              iterations={job.iterations}
              higherIsBetter={job.higher_is_better}
              scoreLabel={job.score_label}
              selectedIteration={selectedIteration}
              onSelect={setSelectedIteration}
            />
          </section>

          <MoleculePanel
            jobId={job.id}
            iteration={selected}
            title="Selected step"
            emptyMessage="Select a scored step from the curve or table."
            showReason
            showScores
            showExplanation
          />

          <section className="panel">
            <h2>Steps</h2>
            {job.iterations.length === 0 ? (
              <p className="muted">No scored steps yet.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Iter</th>
                      <th>SMILES</th>
                      <th>{job.score_label}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {job.iterations.map((step) => {
                      const isBest = job.best?.iteration === step.iteration
                      const isSelected = selectedIteration === step.iteration
                      return (
                        <tr
                          key={step.iteration}
                          className={[
                            isBest ? 'best-row' : '',
                            isSelected ? 'selected-row' : '',
                          ]
                            .filter(Boolean)
                            .join(' ')}
                          onClick={() => setSelectedIteration(step.iteration)}
                        >
                          <td>
                            {step.iteration}
                            {isBest ? ' ★' : ''}
                          </td>
                          <td title={step.smiles}>{shortenSmiles(step.smiles)}</td>
                          <td>{formatScore(step.score)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      ) : null}

      {job.status === 'succeeded' && job.summary ? (
        <section className="panel">
          <h2>Summary</h2>
          <pre className="summary">{job.summary}</pre>
        </section>
      ) : null}
    </div>
  )
}
