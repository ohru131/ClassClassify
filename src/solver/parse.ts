import * as XLSX from 'xlsx-js-style'
import { detectKind } from './columns'
import type { ColumnSpec, Problem, Student } from './types'

type Row = (string | number | null)[]

const cellStr = (v: unknown): string => (v === null || v === undefined ? '' : String(v).trim())

const toNumber = (v: unknown): number | null => {
  const s = cellStr(v)
  if (s === '') return null
  const x = Number(s)
  return Number.isFinite(x) ? x : null
}

function sheetRows(wb: XLSX.WorkBook, name: string): Row[] | null {
  const ws = wb.Sheets[name]
  if (!ws) return null
  return XLSX.utils.sheet_to_json<Row>(ws, { header: 1, defval: null, raw: true, blankrows: true })
}

/** ○/〇/◯ など表記ゆれを統一 */
const normalizeMark = (s: string) => s.replace(/[〇◯○⚪︎]/g, '○')


function readGroups(rows: Row[] | null, noToIndex: Map<number, number>, label: string, warnings: string[]) {
  const groups: number[][] = []
  if (!rows) return groups
  rows.forEach((row, r) => {
    const members: number[] = []
    for (const cell of row) {
      const no = toNumber(cell)
      if (no === null) continue
      const idx = noToIndex.get(Math.trunc(no))
      if (idx === undefined) warnings.push(`「${label}」${r + 1}行目: 出席番号 ${no} は名簿にありません`)
      else if (!members.includes(idx)) members.push(idx)
    }
    if (members.length >= 2) groups.push(members)
  })
  return groups
}

export function parseWorkbook(data: ArrayBuffer): Problem {
  const wb = XLSX.read(data, { type: 'array' })
  const warnings: string[] = []

  const roster = sheetRows(wb, '生徒名簿')
  if (!roster || roster.length < 3) throw new Error('「生徒名簿」シートが見つからないか、データがありません')

  const weightRow = roster[0]
  const headerRow = roster[1].map(cellStr)
  const noCol = headerRow.findIndex((h) => h.toUpperCase().replace(/[.．]/g, '') === 'NO')
  const nameCol = headerRow.findIndex((h) => h === '名前' || h === '氏名')
  const c0 = noCol >= 0 ? noCol : 0
  const c1 = nameCol >= 0 ? nameCol : 1

  const attrCols: { col: number; name: string; weight: number }[] = []
  headerRow.forEach((h, col) => {
    if (col === c0 || col === c1 || h === '') return
    const w = toNumber(weightRow[col])
    attrCols.push({ col, name: h, weight: w !== null && w >= 0 ? w : 1 })
  })

  const students: Student[] = []
  const noToIndex = new Map<number, number>()
  for (let r = 2; r < roster.length; r++) {
    const row = roster[r]
    const rawNo = toNumber(row[c0])
    if (rawNo === null) {
      if (cellStr(row[c1]) !== '') warnings.push(`生徒名簿 ${r + 1}行目: 「${cellStr(row[c1])}」の NO が空欄のため読み込みませんでした`)
      continue
    }
    const no = Math.trunc(rawNo)
    const values: Record<string, string> = {}
    for (const { col, name } of attrCols) values[name] = normalizeMark(cellStr(row[col]))
    if (noToIndex.has(no)) warnings.push(`出席番号 ${no} が重複しています`)
    noToIndex.set(no, students.length)
    students.push({ no, name: cellStr(row[c1]), values })
  }
  if (students.length === 0) throw new Error('生徒データがありません（3行目以降に NO と名前を入力してください）')

  const columns: ColumnSpec[] = attrCols.map(({ name, weight }) => {
    const { kind, levels } = detectKind(students.map((s) => s.values[name]))
    return { name, weight, kind, levels, enabled: levels.length > 0 && weight > 0 }
  })

  const settings = sheetRows(wb, '設定')
  let numClasses = 0
  let maxPerClass: number | null = null
  if (settings) {
    for (const row of settings) {
      const key = cellStr(row[0])
      const val = toNumber(row[1])
      if (val === null) continue
      if (key.includes('クラス数') || key.includes('組数') || key.includes('グループ数')) numClasses = Math.trunc(val)
      else if (key.includes('最大')) maxPerClass = Math.trunc(val)
    }
    if (!numClasses) numClasses = Math.trunc(toNumber(settings[2]?.[1]) ?? 0)
    // クラス数が数式（=CEILING(B1/B2,1) など）で計算結果が保存されていない場合は、最大人数から求める
    if (!numClasses && maxPerClass && maxPerClass > 0) numClasses = Math.ceil(students.length / maxPerClass)
  }
  if (!numClasses || numClasses < 2) {
    numClasses = Math.max(2, Math.round(students.length / 30))
    warnings.push(`「設定」シートのクラス数が読み取れないため ${numClasses} クラスとしました`)
  }
  if (maxPerClass !== null && maxPerClass * numClasses < students.length) {
    warnings.push(`1クラスの最大人数 ${maxPerClass} × ${numClasses} クラスでは全員が収まりません。最大人数を無視します`)
    maxPerClass = null
  }

  const wantedGroups = readGroups(sheetRows(wb, '同じ組ペア'), noToIndex, '同じ組ペア', warnings)
  const unwantedGroups = readGroups(sheetRows(wb, '別の組ペア'), noToIndex, '別の組ペア', warnings)

  return { students, columns, numClasses, maxPerClass, wantedGroups, unwantedGroups, warnings }
}
