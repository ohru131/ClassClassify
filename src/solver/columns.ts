import type { ColumnKind } from './types'

const isNum = (v: string) => v.trim() !== '' && Number.isFinite(Number(v))

/** 「程度」の段階（1〜5の整数） */
export const DEGREE_LEVELS = ['1', '2', '3', '4', '5'] as const
export const isDegreeValue = (v: string) => (DEGREE_LEVELS as readonly string[]).includes(v)

/** 列の値から種類（チェック/リスト/程度/数値）と水準を判定する */
export function detectKind(values: string[]): { kind: ColumnKind; levels: string[] } {
  const levels = [...new Set(values.filter((v) => v !== ''))]
  const allNumeric = levels.length > 0 && levels.every((v) => isNum(v))
  if (allNumeric) levels.sort((a, b) => Number(a) - Number(b))
  else levels.sort((a, b) => a.localeCompare(b, 'ja'))
  if (levels.length <= 1) return { kind: 'flag', levels }
  if (levels.every(isDegreeValue)) return { kind: 'degree', levels }
  if (allNumeric && levels.length > 6) return { kind: 'numeric', levels }
  return { kind: 'category', levels }
}

/**
 * 項目シートに書いてあった種類・選択肢を、値から判定した結果に当てる。
 * 値と合わない種類（数値の列に文字が入っている、など）は値からの判定を使う。
 */
export function withDeclaredKind(detected: { kind: ColumnKind; levels: string[] }, kind: ColumnKind | undefined, options: string[]): { kind: ColumnKind; levels: string[] } {
  const { levels } = detected
  if (kind === 'category') return { kind, levels: [...options, ...levels.filter((l) => !options.includes(l))] }
  if (kind === 'degree' && levels.every(isDegreeValue)) return { kind, levels }
  if (kind === 'numeric' && levels.every(isNum)) return { kind, levels }
  if (kind === 'flag' && levels.length <= 1) return { kind, levels }
  return detected
}
