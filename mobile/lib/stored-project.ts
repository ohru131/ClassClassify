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

export const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null
const isIntArray = (v: unknown): v is number[] => Array.isArray(v) && v.every((x) => Number.isInteger(x))
const isStringArray = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === 'string')
/** クラス数: 2 以上の整数（0.5 や NaN が入るとソルバーと画面が壊れる） */
const isClassCount = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 2

export function isProblem(v: unknown): v is Problem {
  if (!isObj(v)) return false
  const { students, columns, numClasses, maxPerClass, wantedGroups, unwantedGroups, warnings } = v
  if (!Array.isArray(students) || !Array.isArray(columns) || !isStringArray(warnings)) return false
  if (!isClassCount(numClasses) || !(maxPerClass === null || Number.isInteger(maxPerClass))) return false
  const n = students.length
  // values はセル値（文字列）の辞書。数値や null が混ざると集計・表示で落ちる
  const okValues = (v: unknown) => isObj(v) && Object.values(v).every((x) => typeof x === 'string')
  const okStudent = (s: unknown) => isObj(s) && typeof s.no === 'number' && typeof s.name === 'string' && okValues(s.values)
  const okColumn = (c: unknown) =>
    isObj(c) && typeof c.name === 'string' && typeof c.weight === 'number' && typeof c.enabled === 'boolean' && isStringArray(c.levels) && ['flag', 'category', 'numeric'].includes(c.kind as string)
  const okGroups = (g: unknown) => Array.isArray(g) && g.every((x) => isIntArray(x) && x.every((i) => i >= 0 && i < n))
  return students.every(okStudent) && columns.every(okColumn) && okGroups(wantedGroups) && okGroups(unwantedGroups)
}

export function isStoredProject(v: unknown): v is StoredProject {
  if (!isObj(v) || v.version !== 1) return false
  if (!(v.problem === null || isProblem(v.problem))) return false
  if (!(v.fileName === null || typeof v.fileName === 'string')) return false
  if (!isClassCount(v.numClasses) || typeof v.timeSec !== 'number' || !Number.isFinite(v.timeSec)) return false
  const s = v.solution
  if (s === null) return true
  if (!isObj(s) || !isIntArray(s.classOf) || !isIntArray(s.original) || !isClassCount(s.k)) return false
  const k = s.k
  const n = (v.problem as Problem | null)?.students.length ?? -1
  const inRange = (c: number) => c >= 0 && c < k
  // original（最適化直後の割り当て）も同じ範囲。手で動かした差分の表示が範囲外の組を指さないように
  return s.classOf.length === n && s.original.length === n && s.classOf.every(inRange) && s.original.every(inRange)
}
