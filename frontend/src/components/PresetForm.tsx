import {
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
  type FormEvent,
} from 'react'
import { Link } from 'react-router-dom'
import { ApiError, createJob } from '../api'
import type { Preset } from '../types'

type Props = {
  presets: Preset[]
  onCreated: (jobId: string) => void
}

function defaultsFor(preset: Preset): Record<string, string> {
  const values: Record<string, string> = {}
  for (const field of preset.fields) {
    values[field.name] =
      field.default == null ? '' : String(field.default)
  }
  return values
}

function coerceParams(
  preset: Preset,
  values: Record<string, string>,
): Record<string, string | number> {
  const params: Record<string, string | number> = {}
  for (const field of preset.fields) {
    const raw = values[field.name] ?? ''
    if (field.type === 'number') {
      if (raw === '' && !field.required) {
        continue
      }
      params[field.name] = Number(raw)
    } else {
      params[field.name] = raw
    }
  }
  return params
}

export function PresetForm({ presets, onCreated }: Props) {
  const initialId = presets[0]?.id ?? ''
  const [presetId, setPresetId] = useState(initialId)
  const preset = useMemo(
    () => presets.find((item) => item.id === presetId) ?? presets[0],
    [presetId, presets],
  )
  const [values, setValues] = useState<Record<string, string>>(() =>
    preset ? defaultsFor(preset) : {},
  )
  const [error, setError] = useState<string | null>(null)
  const [runningJobId, setRunningJobId] = useState<string | undefined>()
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (!preset) {
      return
    }
    setValues(defaultsFor(preset))
    setError(null)
    setRunningJobId(undefined)
  }, [preset])

  if (!preset) {
    return <p className="muted">No presets available.</p>
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    setRunningJobId(undefined)
    try {
      const result = await createJob(preset.id, coerceParams(preset, values))
      onCreated(result.id)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail)
        setRunningJobId(err.runningJobId)
      } else {
        setError(err instanceof Error ? err.message : 'Failed to start job')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form className="panel form" onSubmit={onSubmit}>
      <label className="field">
        <span>Preset</span>
        <select
          value={preset.id}
          onChange={(event) => setPresetId(event.target.value)}
        >
          {presets.map((item) => (
            <option key={item.id} value={item.id}>
              {item.title}
            </option>
          ))}
        </select>
      </label>

      <p className="muted">{preset.description}</p>

      {preset.id === 'boltz2' ? (
        <p className="note">
          Each Boltz-2 iteration can take several minutes. This job occupies the
          only available slot.
        </p>
      ) : null}

      {preset.fields.map((field) => {
        const value = values[field.name] ?? ''
        const common = {
          id: field.name,
          name: field.name,
          required: field.required,
          value,
          onChange: (
            event: ChangeEvent<
              HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
            >,
          ) =>
            setValues((current) => ({
              ...current,
              [field.name]: event.target.value,
            })),
        }

        return (
          <label key={field.name} className="field" htmlFor={field.name}>
            <span>{field.label}</span>
            {field.type === 'textarea' ? (
              <textarea {...common} rows={field.name === 'protein_sequence' ? 6 : 3} />
            ) : null}
            {field.type === 'text' ? <input type="text" {...common} /> : null}
            {field.type === 'number' ? (
              <input
                type="number"
                {...common}
                min={field.minimum}
                max={field.maximum}
                step="any"
              />
            ) : null}
            {field.type === 'select' ? (
              <select {...common}>
                {(field.options ?? []).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : null}
          </label>
        )
      })}

      {error ? (
        <p className="error" role="alert">
          {error}
          {runningJobId ? (
            <>
              {' '}
              <Link to={`/jobs/${runningJobId}`}>Open running job</Link>
            </>
          ) : null}
        </p>
      ) : null}

      <button type="submit" disabled={submitting}>
        {submitting ? 'Starting…' : 'Start optimization'}
      </button>
    </form>
  )
}
