import type { WorkBook } from 'xlsx-js-style'
import XLSX from './xlsx'
import { detectKind, withDeclaredKind } from './columns'
import type { ColumnSpec, Problem, Student } from './types'
import { isClassCountKey, isClassHeader, isMaxPerClassKey, isNameHeader, isNoHeader, JA_PARSE_MESSAGES, kindFromName, SHEET_ALIASES, type ParseMessages } from './labels'

type Row = (string | number | null)[]

const cellStr = (v: unknown): string => (v === null || v === undefined ? '' : String(v).trim())

const toNumber = (v: unknown): number | null => {
  const s = cellStr(v)
  if (s === '') return null
  const x = Number(s)
  return Number.isFinite(x) ? x : null
}

function sheetRows(wb: WorkBook, name: string): Row[] | null {
  const ws = wb.Sheets[name]
  if (!ws) return null
  return XLSX.utils.sheet_to_json<Row>(ws, { header: 1, defval: null, raw: true, blankrows: true })
}

/** 別名のうち最初に見つかったシート（日本語の名前を最初に探す）。名前も返す（警告に使う） */
function findSheet(wb: WorkBook, aliases: string[]): { rows: Row[] | null; name: string } {
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
 * NO と名前の列（無ければ -1）。まず従来どおりの完全一致（NO / 名前 / 氏名）を列全体から探し、
 * 見つからないときだけ他の言語の見出しを見る。1回の findIndex で両方を見ると、「NO | Name | 氏名」の
 * ようなファイルで左にある Name が勝ってしまい、日本語のファイルの読み方が変わる
 */
function findNoNameCols(headerRow: string[]): { noCol: number; nameCol: number } {
  const findCol = (legacy: (h: string) => boolean, other: (h: string) => boolean) => {
    const i = headerRow.findIndex(legacy)
    return i >= 0 ? i : headerRow.findIndex(other)
  }
  return {
    noCol: findCol((h) => h.toUpperCase().replace(/[.．]/g, '') === 'NO', isNoHeader),
    nameCol: findCol((h) => h === '名前' || h === '氏名', isNameHeader),
  }
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
  const { noCol, nameCol } = findNoNameCols(headerRow)
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

  // 項目シート（このアプリが書き出した名簿にある）: 項目名 → 種類・リストの選択肢
  const declared = new Map<string, { kind: ReturnType<typeof kindFromName>; options: string[] }>()
  for (const row of (findSheet(wb, SHEET_ALIASES.attributes).rows ?? []).slice(1)) {
    const name = cellStr(row[0])
    if (name) declared.set(name, { kind: kindFromName(cellStr(row[1])), options: [...new Set(row.slice(2).map(cellStr).filter((v) => v !== ''))] })
  }
  const columns: ColumnSpec[] = attrCols.map(({ name, weight }) => {
    const d = declared.get(name)
    const { kind, levels } = withDeclaredKind(detectKind(students.map((s) => s.values[name])), d?.kind, d?.options ?? [])
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

/** 書き出した結果から読み取った組分け（各生徒の組の名前） */
export interface Placement {
  students: { no: number; name: string }[]
  /** students と同じ並びの組の名前 */
  classNames: string[]
  /** 組の名前（1組・2組…の順） */
  order: string[]
}

/**
 * このアプリ（Web 版・スマホ版）が書き出した結果の Excel から組分けを読む（「組分け」シートの NO・名前・組）。
 * どの言語で書き出したファイルも読める。Excel として読めない・組分けのシートが無い・NO／名前／組の列が無いときは null
 */
export function parsePlacement(data: ArrayBuffer): Placement | null {
  let wb: WorkBook
  try {
    // 使うのは組分けのシートだけ（各組・集計などのシートは読まない）
    wb = XLSX.read(data, { type: 'array', sheets: SHEET_ALIASES.assign })
  } catch {
    return null
  }
  const rows = findSheet(wb, SHEET_ALIASES.assign).rows
  if (!rows || rows.length < 2) return null
  const header = rows[0].map(cellStr)
  const { noCol, nameCol } = findNoNameCols(header)
  const classCol = header.findIndex(isClassHeader)
  // 照らし合わせは名前が先（名前が無いと NO でも照らさない）ので、名前の列は必須
  if (noCol < 0 || nameCol < 0 || classCol < 0) return null

  const students: Placement['students'] = []
  const classNames: string[] = []
  for (const row of rows.slice(1)) {
    const no = toNumber(row[noCol])
    const cls = cellStr(row[classCol])
    if (no === null || cls === '') continue
    students.push({ no: Math.trunc(no), name: cellStr(row[nameCol]) })
    classNames.push(cls)
  }
  if (students.length === 0) return null
  const order = [...new Set(classNames)].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
  return { students, classNames, order }
}
