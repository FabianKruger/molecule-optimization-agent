import { useEffect, useState } from 'react'
import { fetchHealth } from '../api'
import type { HealthResponse } from '../types'

function label(value: string | undefined): string {
  return value && value.trim() ? value : 'unknown'
}

export function BuildFooter() {
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const data = await fetchHealth()
        if (!cancelled) {
          setHealth(data)
        }
      } catch {
        if (!cancelled) {
          setHealth(null)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  const buildTime = label(health?.build_time)
  const appSha = label(health?.app_sha)
  const deploySha = label(health?.deploy_sha)

  return (
    <footer className="build-footer" aria-label="Build identity">
      <span>
        Build {buildTime} · app {appSha}
      </span>
      <span className="build-footer-deploy" title={`deploy ${deploySha}`}>
        deploy {deploySha}
      </span>
    </footer>
  )
}
