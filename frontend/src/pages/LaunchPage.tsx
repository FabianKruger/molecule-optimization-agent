import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchJobs, fetchPresets } from '../api'
import { JobList } from '../components/JobList'
import { PresetForm } from '../components/PresetForm'
import type { JobSummary, Preset } from '../types'

export function LaunchPage() {
  const navigate = useNavigate()
  const [presets, setPresets] = useState<Preset[] | null>(null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [presetList, jobList] = await Promise.all([
          fetchPresets(),
          fetchJobs(),
        ])
        if (cancelled) {
          return
        }
        setPresets(presetList)
        setJobs(jobList)
        setError(null)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load')
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="page">
      <header className="page-header">
        <h1>seismo</h1>
        <p>Launch an optimization job and inspect its trajectory.</p>
      </header>

      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}

      {presets == null && !error ? <p className="muted">Loading presets…</p> : null}

      {presets ? (
        <section className="stack">
          <h2>New job</h2>
          <PresetForm
            presets={presets}
            onCreated={(jobId) => {
              void navigate(`/jobs/${jobId}`)
            }}
          />
        </section>
      ) : null}

      <section className="stack">
        <h2>Recent jobs</h2>
        <JobList jobs={jobs} />
      </section>
    </div>
  )
}
