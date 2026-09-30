import type { ColumnKind } from './types'

const isNum = (v: string) => v.trim() !== '' && Number.isFinite(Number(v))

/** 列の値から種類（該当/カテゴリ/数値）と水準を判定する */
export function detectKind(values: string[]): { kind: ColumnKind; levels: string[] } {
  const levels = [...new Set(values.filter((v) => v !== ''))]
  const allNumeric = levels.length > 0 && levels.every((v) => isNum(v))
  if (allNumeric) levels.sort((a, b) => Number(a) - Number(b))
  else levels.sort((a, b) => a.localeCompare(b, 'ja'))
  if (levels.length <= 1) return { kind: 'flag', levels }
  if (allNumeric && levels.length > 6) return { kind: 'numeric', levels }
  return { kind: 'category', levels }
}
