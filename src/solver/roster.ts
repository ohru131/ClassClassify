import { detectKind } from './columns'
import type { ColumnKind, ColumnSpec, Problem, Student } from './types'

export type GroupKind = 'wanted' | 'unwanted'
const key = (k: GroupKind) => (k === 'wanted' ? 'wantedGroups' : 'unwantedGroups')

/**
 * 生徒の値から各列の水準を再計算する（重み・有効/無効は維持）。
 * 編集中に入力欄の種類が変わらないよう、数値・カテゴリ列の種類は保持する（該当→カテゴリへの昇格のみ）。
 */
export function refreshColumns(students: Student[], columns: ColumnSpec[]): ColumnSpec[] {
  return columns.map((c) => {
    const detected = detectKind(students.map((s) => s.values[c.name] ?? ''))
    const { levels } = detected
    const allNumeric = levels.every((v) => Number.isFinite(Number(v)))
    const kind: ColumnKind =
      c.kind === 'numeric' && allNumeric ? 'numeric' : c.kind === 'category' || (c.kind === 'numeric' && !allNumeric) ? 'category' : detected.kind
    const wasEmpty = c.levels.length === 0
    return { ...c, kind, levels, enabled: wasEmpty && levels.length > 0 ? c.weight > 0 : c.enabled && levels.length > 0 }
  })
}

const withStudents = (p: Problem, students: Student[]): Problem => ({ ...p, students, columns: refreshColumns(students, p.columns) })

export function updateStudent(p: Problem, i: number, patch: { name?: string; no?: number; values?: Record<string, string> }): Problem {
  const students = p.students.map((s, j) =>
    j === i ? { ...s, ...(patch.name !== undefined && { name: patch.name }), ...(patch.no !== undefined && { no: patch.no }), values: { ...s.values, ...patch.values } } : s,
  )
  return patch.values ? withStudents(p, students) : { ...p, students }
}

/** 複数の生徒の同じ項目をまとめて変更 */
export function setValueFor(p: Problem, idxs: number[], column: string, value: string): Problem {
  const set = new Set(idxs)
  return withStudents(
    p,
    p.students.map((s, i) => (set.has(i) ? { ...s, values: { ...s.values, [column]: value } } : s)),
  )
}

export function addStudent(p: Problem): Problem {
  const no = p.students.reduce((m, s) => Math.max(m, s.no), 0) + 1
  const values = Object.fromEntries(p.columns.map((c) => [c.name, '']))
  return { ...p, students: [...p.students, { no, name: '', values }] }
}

/** 生徒を削除し、ペア指定の index を詰め直す（2人未満になったグループは消す） */
export function removeStudents(p: Problem, idxs: number[]): Problem {
  const del = new Set(idxs)
  const map = new Map<number, number>()
  p.students.forEach((_, i) => {
    if (!del.has(i)) map.set(i, map.size)
  })
  const remap = (groups: number[][]) =>
    groups.map((g) => g.filter((i) => map.has(i)).map((i) => map.get(i)!)).filter((g) => g.length >= 2)
  return {
    ...withStudents(
      p,
      p.students.filter((_, i) => !del.has(i)),
    ),
    wantedGroups: remap(p.wantedGroups),
    unwantedGroups: remap(p.unwantedGroups),
  }
}

export function addColumn(p: Problem, name: string, kind: ColumnKind = 'flag'): Problem {
  if (!name || name === 'NO' || name === '名前' || p.columns.some((c) => c.name === name)) return p
  return {
    ...p,
    students: p.students.map((s) => ({ ...s, values: { ...s.values, [name]: '' } })),
    columns: [...p.columns, { name, weight: 1, kind, levels: [], enabled: false }],
  }
}

/** NO の重複チェック（i 番目の生徒を除く） */
export const isNoTaken = (p: Problem, no: number, except: number) => p.students.some((s, i) => i !== except && s.no === no)

export function removeColumn(p: Problem, name: string): Problem {
  return {
    ...p,
    students: p.students.map((s) => {
      const values = { ...s.values }
      delete values[name]
      return { ...s, values }
    }),
    columns: p.columns.filter((c) => c.name !== name),
  }
}

const uniq = (xs: number[]) => [...new Set(xs)]

export function addGroup(p: Problem, kind: GroupKind, members: number[]): Problem {
  const g = uniq(members)
  if (g.length < 2) return p
  return { ...p, [key(kind)]: [...p[key(kind)], g] }
}

export function setGroup(p: Problem, kind: GroupKind, gi: number, members: number[]): Problem {
  const g = uniq(members)
  const groups = p[key(kind)]
  return { ...p, [key(kind)]: g.length >= 2 ? groups.map((x, i) => (i === gi ? g : x)) : groups.filter((_, i) => i !== gi) }
}

export function removeGroup(p: Problem, kind: GroupKind, gi: number): Problem {
  return { ...p, [key(kind)]: p[key(kind)].filter((_, i) => i !== gi) }
}

/** 生徒ごとの所属グループ */
export function groupsOf(p: Problem) {
  const wanted = p.students.map(() => [] as number[])
  const unwanted = p.students.map(() => [] as number[])
  p.wantedGroups.forEach((g, gi) => g.forEach((i) => wanted[i]?.push(gi)))
  p.unwantedGroups.forEach((g, gi) => g.forEach((i) => unwanted[i]?.push(gi)))
  return { wanted, unwanted }
}

/**
 * 矛盾する指定を検出: 「同じ組」で（連鎖的に）つながる2人が「別の組」にも指定されている
 * 例: A-B 同じ組、B-C 同じ組、A-C 別の組
 */
export function findConflicts(p: Problem): [number, number][] {
  const parent = p.students.map((_, i) => i)
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])))
  for (const g of p.wantedGroups) for (const i of g.slice(1)) parent[find(i)] = find(g[0])
  const out: [number, number][] = []
  for (const g of p.unwantedGroups)
    for (let a = 0; a < g.length; a++) for (let b = a + 1; b < g.length; b++) if (find(g[a]) === find(g[b])) out.push([g[a], g[b]])
  return out
}

/** 数値として正規な文字列（"3", "2.5"）だけ数値にする。"007" などは文字列のまま */
export const toCell = (v: string | undefined): string | number =>
  v === undefined || v === '' ? '' : String(Number(v)) === v ? Number(v) : v

/** 出力用の元名簿シートを現在の内容から作り直す */
export function rosterRows(p: Problem): (string | number | null)[][] {
  return [
    // 無効にした項目は重み 0 で保存（再読み込み時も無効のまま）
    ['', '重み', ...p.columns.map((c) => (c.enabled ? c.weight : 0))],
    ['NO', '名前', ...p.columns.map((c) => c.name)],
    ...p.students.map((s) => [s.no, s.name, ...p.columns.map((c) => toCell(s.values[c.name]))]),
  ]
}
