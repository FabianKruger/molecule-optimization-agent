import type {
  ApiErrorBody,
  CreateJobResponse,
  JobDetail,
  JobSummary,
  Preset,
} from './types'

export class ApiError extends Error {
  status: number
  detail: string
  runningJobId?: string

  constructor(status: number, detail: string, runningJobId?: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.runningJobId = runningJobId
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') {
    return detail
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === 'string') {
          return item
        }
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg: unknown }).msg)
        }
        return JSON.stringify(item)
      })
      .join('; ')
  }
  if (detail == null) {
    return 'Request failed'
  }
  return JSON.stringify(detail)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
    ...init,
  })

  if (!response.ok) {
    let body: ApiErrorBody = {}
    try {
      body = (await response.json()) as ApiErrorBody
    } catch {
      body = {}
    }
    throw new ApiError(
      response.status,
      formatDetail(body.detail) || response.statusText || 'Request failed',
      body.running_job_id,
    )
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export async function fetchPresets(): Promise<Preset[]> {
  const data = await request<{ presets: Preset[] }>('/api/presets')
  return data.presets
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const data = await request<{ jobs: JobSummary[] }>('/api/jobs')
  return data.jobs
}

export async function fetchJob(id: string): Promise<JobDetail> {
  return request<JobDetail>(`/api/jobs/${id}`)
}

export async function createJob(
  presetId: string,
  params: Record<string, string | number>,
): Promise<CreateJobResponse> {
  return request<CreateJobResponse>('/api/jobs', {
    method: 'POST',
    body: JSON.stringify({ preset_id: presetId, params }),
  })
}

export async function cancelJob(id: string): Promise<CreateJobResponse> {
  return request<CreateJobResponse>(`/api/jobs/${id}/cancel`, {
    method: 'POST',
  })
}

export async function previewFirstMessage(
  presetId: string,
  params: Record<string, string | number>,
): Promise<string> {
  const data = await request<{ first_message: string }>(
    `/api/presets/${encodeURIComponent(presetId)}/first-message`,
    {
      method: 'POST',
      body: JSON.stringify({ params }),
    },
  )
  return data.first_message
}

export function structureUrl(jobId: string, iteration: number): string {
  return `/api/jobs/${jobId}/iterations/${iteration}/structure`
}
