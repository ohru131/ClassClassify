import { COPY } from './copy'
import type { AppLanguage } from './i18n'
import { LANGUAGE_META } from './i18n'
import type { ColumnSpec, Problem } from './solver'
import { isProblem } from './stored-project'

// 名前を付けて端末に保存した編成結果（あとで一覧から開く・削除する・「前回の組」の元にする）。
// React・AsyncStorage に依存しない純関数だけを置く（テストから使う）。保存先は saved-results-store.tsx。

export interface SavedResult {
  version: 1
  id: string
  name: string
  /** ISO 8601 */
  savedAt: string
  problem: Problem
  classOf: number[]
  k: number
}

/** 一覧に出す分だけ（本体を全部読まずに一覧を出すため、別のキーにまとめて置く） */
export interface SavedMeta {
  id: string
  name: string
  savedAt: string
  n: number
  k: number
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null

export function isSavedResult(v: unknown): v is SavedResult {
  if (!isObj(v) || v.version !== 1) return false
  if (typeof v.id !== 'string' || typeof v.name !== 'string' || typeof v.savedAt !== 'string') return false
  if (!isProblem(v.problem) || !Number.isInteger(v.k) || (v.k as number) < 2) return false
  const k = v.k as number
  const { classOf } = v
  return Array.isArray(classOf) && classOf.length === v.problem.students.length && classOf.every((c) => Number.isInteger(c) && c >= 0 && c < k)
}

export function isSavedMetaList(v: unknown): v is SavedMeta[] {
  return (
    Array.isArray(v) &&
    v.every((m) => isObj(m) && typeof m.id === 'string' && typeof m.name === 'string' && typeof m.savedAt === 'string' && Number.isInteger(m.n) && Number.isInteger(m.k))
  )
}

export const metaOf = (s: SavedResult): SavedMeta => ({ id: s.id, name: s.name, savedAt: s.savedAt, n: s.problem.students.length, k: s.k })

/** 「2026年10月」「October 2026」のような年月（Intl が使えなければ 2026-10） */
export function yearMonth(lang: AppLanguage, d: Date): string {
  try {
    return new Intl.DateTimeFormat(LANGUAGE_META[lang].intl, { year: 'numeric', month: 'long' }).format(d)
  } catch {
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
  }
}

/** 保存名の既定値: 年月 · 名簿の名前（一覧で「いつの・どの名簿か」が分かるように） */
export function defaultSaveName(lang: AppLanguage, d: Date, rosterName: string | null): string {
  const ym = yearMonth(lang, d)
  // 保存した編成を開いて保存し直すとき、前の年月を重ねない（「2026年3月 · 2026年3月 · …」にしない）
  const name = rosterName?.replace(/\.xlsx$/i, '').replace(/^[^·]*\d{4}[^·]*·\s*/, '').trim()
  return name ? `${ym} · ${name}` : ym
}

const norm = (s: string) => s.replace(/\s+/g, '').normalize('NFKC')

/**
 * 今の名簿の各生徒について、保存した編成で何組だったか（組の名前）。見つからない生徒は ''。
 * 名前で照らし合わせ（同じ名前が2人以上いる名前は使わない）、名前で見つからなければ NO で照らす。
 */
export function previousClassValues(current: Problem, saved: SavedResult, className: (c: number) => string): { values: string[]; matched: number } {
  const byName = new Map<string, number | null>()
  saved.problem.students.forEach((s, i) => {
    const key = norm(s.name)
    if (!key) return
    byName.set(key, byName.has(key) ? null : i)
  })
  const byNo = new Map<number, number | null>()
  saved.problem.students.forEach((s, i) => byNo.set(s.no, byNo.has(s.no) ? null : i))
  let matched = 0
  const values = current.students.map((s) => {
    const i = byName.get(norm(s.name)) ?? byNo.get(s.no) ?? null
    if (i === null) return ''
    matched++
    return className(saved.classOf[i])
  })
  return { values, matched }
}

/**
 * 名簿にある「前回の組」の列（どの言語で入れたものでも見つける）。表示の言語を切り替えたあとに
 * 別の名前でもう1本入れたり、古い列を残したまま有効にしたりしないように。
 */
const PREVIOUS_CLASS_NAMES = new Set(Object.values(COPY).map((c) => c.prevClassColumn))
export const findPreviousClassColumn = (p: Problem): ColumnSpec | undefined => p.columns.find((c) => PREVIOUS_CLASS_NAMES.has(c.name))

/** 「前回の組」の重み（他の項目より優先して散らす） */
export const PREVIOUS_CLASS_WEIGHT = 2

/**
 * 名簿に「前回の組」の列を入れる（同じ名前の列があれば置き換える）。カテゴリとして均等に散らすと、
 * 新しい組の中で前回同じ組だった子の組み合わせが最も少なくなる（組ごとの人数を均すのと同じこと）。
 */
export function withPreviousClass(p: Problem, column: string, values: string[], levelOrder: string[]): Problem {
  const present = new Set(values.filter((v) => v !== ''))
  const levels = levelOrder.filter((l) => present.has(l))
  const old = p.columns.find((c) => c.name === column)
  const spec: ColumnSpec = { name: column, kind: 'category', levels, weight: old && old.weight > 0 ? old.weight : PREVIOUS_CLASS_WEIGHT, enabled: true }
  const columns = old ? p.columns.map((c) => (c.name === column ? spec : c)) : [...p.columns, spec]
  return { ...p, columns, students: p.students.map((s, i) => ({ ...s, values: { ...s.values, [column]: values[i] ?? '' } })) }
}

export function setColumnEnabled(p: Problem, column: string, enabled: boolean): Problem {
  return { ...p, columns: p.columns.map((c) => (c.name === column ? { ...c, enabled } : c)) }
}

/** 無料版で保存しておける件数（Pro は無制限）。開く・削除・「前回の組」に使うのは件数に関係なく無料 */
export const FREE_SAVE_LIMIT = 3

export const canSaveMore = (isPro: boolean, count: number) => isPro || count < FREE_SAVE_LIMIT

export const newSavedId = (d: Date) => `${d.getTime().toString(36)}${Math.random().toString(36).slice(2, 8)}`
