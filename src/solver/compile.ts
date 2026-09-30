import type { CompiledProblem, Problem } from './types'

export interface AttributeMeta {
  column: string
  /** category/flag の水準名。numeric は null */
  level: string | null
}

const pairsOf = (groups: number[][]) => {
  const out: [number, number][] = []
  const seen = new Set<string>()
  for (const g of groups)
    for (let a = 0; a < g.length; a++)
      for (let b = a + 1; b < g.length; b++) {
        const [i, j] = g[a] < g[b] ? [g[a], g[b]] : [g[b], g[a]]
        const key = `${i}-${j}`
        if (!seen.has(key)) {
          seen.add(key)
          out.push([i, j])
        }
      }
  return out
}

export function compile(p: Problem, numClasses = p.numClasses): { compiled: CompiledProblem; meta: AttributeMeta[] } {
  const n = p.students.length
  const k = numClasses
  const attr: number[][] = []
  const weights: number[] = []
  const lo: number[] = []
  const hi: number[] = []
  const target: number[] = []
  const meta: AttributeMeta[] = []

  for (const col of p.columns) {
    if (!col.enabled || col.weight <= 0) continue
    if (col.kind === 'numeric') {
      const raw = p.students.map((s) => (s.values[col.name] === '' ? NaN : Number(s.values[col.name])))
      const valid = raw.filter((x) => Number.isFinite(x))
      if (valid.length === 0) continue
      const mean = valid.reduce((a, b) => a + b, 0) / valid.length
      const sd = Math.sqrt(valid.reduce((a, b) => a + (b - mean) ** 2, 0) / valid.length) || 1
      // 標準化（欠損は平均扱い）→ クラス合計を 0 に近づける
      attr.push(raw.map((x) => (Number.isFinite(x) ? (x - mean) / sd : 0)))
      weights.push(col.weight)
      lo.push(0)
      hi.push(0)
      target.push(0)
      meta.push({ column: col.name, level: null })
    } else {
      for (const level of col.levels) {
        const v: number[] = p.students.map((s) => (s.values[col.name] === level ? 1 : 0))
        const total = v.reduce((a, b) => a + b, 0)
        if (total === 0) continue
        attr.push(v)
        weights.push(col.weight)
        lo.push(Math.floor(total / k))
        hi.push(Math.ceil(total / k))
        target.push(total / k)
        meta.push({ column: col.name, level })
      }
    }
  }

  return {
    compiled: {
      n,
      k,
      minSize: Math.floor(n / k),
      // 人数は均等（差1以内）を前提とする。maxPerClass は読み込み時に ceil(n/k) 以上であることを検証済み
      maxSize: Math.ceil(n / k),
      weights,
      lo,
      hi,
      target,
      attr,
      wantedPairs: pairsOf(p.wantedGroups),
      unwantedPairs: pairsOf(p.unwantedGroups),
    },
    meta,
  }
}
