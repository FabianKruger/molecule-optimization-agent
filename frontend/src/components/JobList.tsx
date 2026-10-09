import { Link } from 'react-router-dom'
import { formatScore, formatTimestamp } from '../format'
import type { JobSummary } from '../types'

type Props = {
  jobs: JobSummary[]
}

function statusLabel(job: JobSummary): string {
  if (job.status === 'queued' && job.queue_position != null) {
    return `queued · #${job.queue_position}`
  }
  return job.status
}

export function JobList({ jobs }: Props) {
  if (jobs.length === 0) {
    return <p className="muted">No jobs yet.</p>
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Preset</th>
            <th>Status</th>
            <th>Created</th>
            <th>Iterations</th>
            <th>Best score</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td>
                <Link to={`/jobs/${job.id}`}>{job.preset_title}</Link>
              </td>
              <td>
                <span className={`status status-${job.status}`}>
                  {statusLabel(job)}
                </span>
              </td>
              <td>{formatTimestamp(job.created_at)}</td>
              <td>
                {job.iterations_completed}/{job.max_iterations}
              </td>
              <td>{formatScore(job.best_score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
