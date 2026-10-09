type Props = {
  message: string | null | undefined
  loading?: boolean
  error?: string | null
  onToggle?: (open: boolean) => void
}

export function FirstMessageDisclosure({
  message,
  loading = false,
  error = null,
  onToggle,
}: Props) {
  const hasMessage = Boolean(message?.trim())

  return (
    <details
      className="first-message"
      onToggle={(event) => {
        onToggle?.(event.currentTarget.open)
      }}
    >
      <summary>Initial instructions</summary>
      {loading ? <p className="muted">Loading…</p> : null}
      {!loading && error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      {!loading && !error && hasMessage ? (
        <pre className="explanation">{message}</pre>
      ) : null}
      {!loading && !error && !hasMessage ? (
        <p className="muted">No initial instructions available for this job.</p>
      ) : null}
    </details>
  )
}
