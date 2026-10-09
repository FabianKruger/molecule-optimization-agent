export function shortenSmiles(smiles: string, max = 40): string {
  if (smiles.length <= max) {
    return smiles
  }
  return `${smiles.slice(0, max - 1)}…`
}

export function formatScore(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) {
    return '—'
  }
  return Number.isInteger(score) ? String(score) : score.toFixed(4)
}

export function formatTimestamp(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  return date.toLocaleString()
}
