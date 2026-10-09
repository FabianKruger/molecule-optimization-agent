export type FieldType = 'text' | 'textarea' | 'number' | 'select'

export type SelectOption = {
  value: string
  label: string
}

export type PresetField = {
  name: string
  label: string
  type: FieldType
  default: string | number | null
  required: boolean
  minimum?: number
  maximum?: number
  options?: SelectOption[]
}

export type Preset = {
  id: string
  title: string
  description: string
  higher_is_better: boolean
  score_label: string
  fields: PresetField[]
}

export type JobStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
export type JobPhase = 'generating' | 'scoring' | 'summarizing' | null

export type JobSummary = {
  id: string
  preset_id: string
  preset_title: string
  status: JobStatus
  created_at: string
  updated_at: string
  iterations_completed: number
  max_iterations: number
  best_score: number | null
  best_smiles: string | null
  queue_position: number | null
}

export type JobBest = {
  iteration: number
  smiles: string
  score: number
}

export type JobIteration = {
  iteration: number
  smiles: string
  reason: string
  score: number | null
  scores?: Record<string, number | null>
  explanation: string
  error?: string
}

export type JobDetail = JobSummary & {
  phase: JobPhase
  error: string | null
  params: Record<string, unknown>
  higher_is_better: boolean
  score_label: string
  summary: string | null
  best: JobBest | null
  iterations: JobIteration[]
  /** Snapshot of objective.first_message() at job start; null on older jobs. */
  first_message: string | null
}

export type CreateJobResponse = {
  id: string
}

export type ApiErrorBody = {
  detail?: unknown
}
