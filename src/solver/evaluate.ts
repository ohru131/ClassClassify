import type { Problem } from './types'

export interface ColumnReport {
  column: string
  kind: 'flag' | 'category' | 'numeric'
  weight: number
  /** category/flag: rows[level][class] = 人数、numeric: rows[0][class] = 平均 */
  levels: string[]
  rows: number[][]
  /** 各水準の理想値（total / クラス数） */
  ideal: number[]
  /** 理想の整数帯（floor〜ceil）からはみ出した人数の合計 */
  excess: number
}

export interface Violation {
  type: 'wanted' | 'unwanted'
  students: number[]
  message: string
}

export interface Report {
  sizes: number[]
  columns: ColumnReport[]
  violations: Violation[]
  /** 属性の帯外れの合計（0 なら全属性が理想の整数範囲内） */
  totalExcess: number
}

const label = (p: Problem, i: number) => `${p.students[i].no}:${p.students[i].name}`

export function evaluate(p: Problem, classOf: number[], k: number): Report {
  const sizes = new Array(k).fill(0)
  for (const c of classOf) sizes[c]++

  const columns: ColumnReport[] = []
  let totalExcess = 0
  for (const col of p.columns) {
    if (!col.enabled || col.levels.length === 0) continue
    if (col.kind === 'numeric') {
      const sum = new Array(k).fill(0)
      const cnt = new Array(k).fill(0)
      let all = 0
      let allN = 0
      p.students.forEach((s, i) => {
        const v = Number(s.values[col.name])
        if (s.values[col.name] === '' || !Number.isFinite(v)) return
        sum[classOf[i]] += v
        cnt[classOf[i]]++
        all += v
        allN++
      })
      columns.push({
        column: col.name,
        kind: col.kind,
        weight: col.weight,
        levels: ['平均'],
        rows: [sum.map((s, c) => (cnt[c] ? s / cnt[c] : 0))],
        ideal: [allN ? all / allN : 0],
        excess: 0,
      })
      continue
    }
    const rows = col.levels.map(() => new Array(k).fill(0))
    p.students.forEach((s, i) => {
      const l = col.levels.indexOf(s.values[col.name])
      if (l >= 0) rows[l][classOf[i]]++
    })
    let excess = 0
    const ideal = rows.map((r) => {
      const total = r.reduce((a, b) => a + b, 0)
      const lo = Math.floor(total / k)
      const hi = Math.ceil(total / k)
      for (const c of r) excess += c > hi ? c - hi : c < lo ? lo - c : 0
      return total / k
    })
    // 帯外れは「移動が必要な人数」相当に揃えるため半分にする
    excess /= 2
    totalExcess += excess * (col.weight > 0 ? 1 : 0)
    columns.push({ column: col.name, kind: col.kind, weight: col.weight, levels: col.levels, rows, ideal, excess })
  }

  const violations: Violation[] = []
  for (const g of p.wantedGroups) {
    if (new Set(g.map((i) => classOf[i])).size > 1)
      violations.push({
        type: 'wanted',
        students: g,
        message: `同じ組にしたい ${g.map((i) => label(p, i)).join('・')} が別の組になっています`,
      })
  }
  for (const g of p.unwantedGroups)
    for (let a = 0; a < g.length; a++)
      for (let b = a + 1; b < g.length; b++)
        if (classOf[g[a]] === classOf[g[b]])
          violations.push({
            type: 'unwanted',
            students: [g[a], g[b]],
            message: `別の組にしたい ${label(p, g[a])} と ${label(p, g[b])} が同じ ${classOf[g[a]] + 1}組 です`,
          })

  return { sizes, columns, violations, totalExcess }
}
