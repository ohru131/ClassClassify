import * as XLSX from 'xlsx-js-style'
import { detectKind } from './columns'
import type { ColumnSpec, Problem, Student } from './types'
import { isClassCountKey, isMaxPerClassKey, isNameHeader, isNoHeader, JA_PARSE_MESSAGES, SHEET_ALIASES, type ParseMessages } from './labels'

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

/** 別名のうち最初に見つかったシート（日本語の名前を最初に探す）。名前も返す（警告に使う） */
function findSheet(wb: XLSX.WorkBook, aliases: string[]): { rows: Row[] | null; name: string } {
  for (const a of aliases) if (wb.Sheets[a]) return { rows: sheetRows(wb, a), name: a }
  return { rows: null, name: aliases[0] }
}

/** ○/〇/◯ など表記ゆれを統一 */
const normalizeMark = (s: string) => s.replace(/[〇◯○⚪︎]/g, '○')


function readGroups(rows: Row[] | null, noToIndex: Map<number, number>, label: string, warnings: string[], msg: ParseMessages) {
  const groups: number[][] = []
  if (!rows) return groups
  rows.forEach((row, r) => {
    const members: number[] = []
    for (const cell of row) {
      const no = toNumber(cell)
      if (no === null) continue
      const idx = noToIndex.get(Math.trunc(no))
      if (idx === undefined) warnings.push(msg.unknownNo(label, r + 1, no))
      else if (!members.includes(idx)) members.push(idx)
    }
    if (members.length >= 2) groups.push(members)
  })
  return groups
}

/**
 * msg: 警告・エラーの文言（既定は日本語）。シート名・見出しは全言語の別名を受け付ける
 * （src/solver/labels.ts。日本語の名前を最初に探すので、日本語のファイルの読み方は変わらない）。
 */
export function parseWorkbook(data: ArrayBuffer, msg: ParseMessages = JA_PARSE_MESSAGES): Problem {
  const wb = XLSX.read(data, { type: 'array' })
  const warnings: string[] = []

  const roster = findSheet(wb, SHEET_ALIASES.roster).rows
  if (!roster || roster.length < 3) throw new Error(msg.rosterMissing)

  const weightRow = roster[0]
  const headerRow = roster[1].map(cellStr)
  const noCol = headerRow.findIndex((h) => h.toUpperCase().replace(/[.．]/g, '') === 'NO' || isNoHeader(h))
  const nameCol = headerRow.findIndex((h) => h === '名前' || h === '氏名' || isNameHeader(h))
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
      if (cellStr(row[c1]) !== '') warnings.push(msg.noMissing(r + 1, cellStr(row[c1])))
      continue
    }
    const no = Math.trunc(rawNo)
    const values: Record<string, string> = {}
    for (const { col, name } of attrCols) values[name] = normalizeMark(cellStr(row[col]))
    if (noToIndex.has(no)) warnings.push(msg.duplicateNo(no))
    noToIndex.set(no, students.length)
    students.push({ no, name: cellStr(row[c1]), values })
  }
  if (students.length === 0) throw new Error(msg.noStudents)

  const columns: ColumnSpec[] = attrCols.map(({ name, weight }) => {
    const { kind, levels } = detectKind(students.map((s) => s.values[name]))
    return { name, weight, kind, levels, enabled: levels.length > 0 && weight > 0 }
  })

  const settings = findSheet(wb, SHEET_ALIASES.settings).rows
  let numClasses = 0
  let maxPerClass: number | null = null
  if (settings) {
    for (const row of settings) {
      const key = cellStr(row[0])
      const val = toNumber(row[1])
      if (val === null) continue
      if (key.includes('クラス数') || key.includes('組数') || key.includes('グループ数') || isClassCountKey(key)) numClasses = Math.trunc(val)
      else if (key.includes('最大') || isMaxPerClassKey(key)) maxPerClass = Math.trunc(val)
    }
    if (!numClasses) numClasses = Math.trunc(toNumber(settings[2]?.[1]) ?? 0)
    // クラス数が数式（=CEILING(B1/B2,1) など）で計算結果が保存されていない場合は、最大人数から求める
    if (!numClasses && maxPerClass && maxPerClass > 0) numClasses = Math.ceil(students.length / maxPerClass)
  }
  if (!numClasses || numClasses < 2) {
    numClasses = Math.max(2, Math.round(students.length / 30))
    warnings.push(msg.classCountUnknown(numClasses))
  }
  if (maxPerClass !== null && maxPerClass * numClasses < students.length) {
    warnings.push(msg.maxTooSmall(maxPerClass, numClasses))
    maxPerClass = null
  }

  const wanted = findSheet(wb, SHEET_ALIASES.wanted)
  const unwanted = findSheet(wb, SHEET_ALIASES.unwanted)
  const wantedGroups = readGroups(wanted.rows, noToIndex, wanted.name, warnings, msg)
  const unwantedGroups = readGroups(unwanted.rows, noToIndex, unwanted.name, warnings, msg)

  return { students, columns, numClasses, maxPerClass, wantedGroups, unwantedGroups, warnings }
}
