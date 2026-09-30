import type { Problem } from './solver'

// 端末に保存したプロジェクトの形。読み戻すときに形を確かめ、壊れていれば捨てる
// （壊れたデータで画面が落ちるより、空から始める方がまし）。
export interface StoredProject {
  version: 1
  problem: Problem | null
  fileName: string | null
  numClasses: number
  timeSec: number
  solution: { classOf: number[]; original: number[]; k: number; iterations: number; starts: number } | null
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null
const isIntArray = (v: unknown): v is number[] => Array.isArray(v) && v.every((x) => Number.isInteger(x))

export function isProblem(v: unknown): v is Problem {
  if (!isObj(v)) return false
  const { students, columns, numClasses, maxPerClass, wantedGroups, unwantedGroups, warnings } = v
  if (!Array.isArray(students) || !Array.isArray(columns) || !Array.isArray(warnings)) return false
  if (typeof numClasses !== 'number' || !(maxPerClass === null || typeof maxPerClass === 'number')) return false
  const n = students.length
  const okStudent = (s: unknown) => isObj(s) && typeof s.no === 'number' && typeof s.name === 'string' && isObj(s.values)
  const okColumn = (c: unknown) =>
    isObj(c) && typeof c.name === 'string' && typeof c.weight === 'number' && typeof c.enabled === 'boolean' && Array.isArray(c.levels) && ['flag', 'category', 'numeric'].includes(c.kind as string)
  const okGroups = (g: unknown) => Array.isArray(g) && g.every((x) => isIntArray(x) && x.every((i) => i >= 0 && i < n))
  return students.every(okStudent) && columns.every(okColumn) && okGroups(wantedGroups) && okGroups(unwantedGroups)
}

export function isStoredProject(v: unknown): v is StoredProject {
  if (!isObj(v) || v.version !== 1) return false
  if (!(v.problem === null || isProblem(v.problem))) return false
  if (!(v.fileName === null || typeof v.fileName === 'string')) return false
  if (typeof v.numClasses !== 'number' || v.numClasses < 2 || typeof v.timeSec !== 'number') return false
  const s = v.solution
  if (s === null) return true
  if (!isObj(s) || !isIntArray(s.classOf) || !isIntArray(s.original) || typeof s.k !== 'number') return false
  const n = (v.problem as Problem | null)?.students.length ?? -1
  return s.classOf.length === n && s.original.length === n && s.classOf.every((c) => c >= 0 && c < (s.k as number))
}
