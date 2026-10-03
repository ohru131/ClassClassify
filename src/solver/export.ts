import type { WorkBook } from 'xlsx-js-style'
import XLSX from './xlsx'
import type { Problem } from './types'
import type { Report } from './evaluate'
import { attributeRows, rosterRows, toCell } from './roster'
import { pairStatus, rowColor, tagText, UNWANTED_COLOR, VIOLATION_COLOR, type PairTag } from './pairs'
import { FILE_LABELS, violationText, type FileLanguage } from './labels'

export type Cell = string | number | null

/** セルの塗り（行 row、列 col から cols 列ぶん） */
export interface Fill {
  row: number
  col: number
  cols?: number
  bg?: string
  fg?: string
  bold?: boolean
}

export interface SheetData {
  name: string
  rows: Cell[][]
  fills?: Fill[]
  /** 列幅（文字数） */
  widths?: number[]
}

const round2 = (v: number) => Math.round(v * 100) / 100
const HEADER: Omit<Fill, 'row' | 'col'> = { bg: '#F1F5F9', bold: true }

/** ペア指定に応じた塗り: 同じ組 → グループ色、別の組のみ → ペア指定セルを薄い赤、未達成 → 赤字 */
function pairFills(tags: PairTag[], row: number, nameCols: [number, number], tagCol: number): Fill[] {
  const out: Fill[] = []
  const color = rowColor(tags)
  if (color) out.push({ row, col: nameCols[0], cols: nameCols[1] - nameCols[0] + 1, bg: color.bg })
  if (tags.length) {
    const bad = tags.some((t) => !t.ok)
    out.push({
      row,
      col: tagCol,
      bg: bad ? VIOLATION_COLOR.bg : (color ?? UNWANTED_COLOR).bg,
      fg: bad ? VIOLATION_COLOR.fg : (color ?? UNWANTED_COLOR).fg,
      bold: true,
    })
  }
  return out
}

/** 結果の各シート（組分け・クラス別名簿・ペア指定・集計・組み合わせ失敗）の中身を作る */
/** lang: シート名・見出しの言語（既定は日本語。Web 版はこの既定のまま） */
export function buildResultSheets(p: Problem, classOf: number[], k: number, report: Report, lang: FileLanguage = 'ja'): SheetData[] {
  const L = FILE_LABELS[lang]
  const className = L.className
  const cols = p.columns.map((c) => c.name)
  const { tags, groups } = pairStatus(p, classOf, L.tagPrefix)
  const byClass = Array.from({ length: k }, (_, c) => p.students.map((_, i) => i).filter((i) => classOf[i] === c))

  // 組分け（全員）
  const assign: SheetData = {
    name: L.sheets.assign,
    rows: [[L.no, L.name, L.classCol, L.pairCol, ...cols]],
    fills: [{ row: 0, col: 0, cols: 4 + cols.length, ...HEADER }],
    widths: [6, 14, 6, 12, ...cols.map(() => 10)],
  }
  p.students.forEach((s, i) => {
    assign.rows.push([s.no, s.name, className(classOf[i]), tagText(tags[i]), ...cols.map((c) => toCell(s.values[c]))])
    assign.fills!.push(...pairFills(tags[i], i + 1, [0, 2], 3))
  })

  // クラス別名簿（横並び: NO / 名前 / 指定）
  const side: SheetData = {
    name: L.sheets.byClass,
    rows: [byClass.flatMap((_, c) => [className(c), '', ''])],
    fills: [{ row: 0, col: 0, cols: k * 3, ...HEADER }],
    widths: byClass.flatMap(() => [6, 14, 10]),
  }
  const height = Math.max(0, ...byClass.map((m) => m.length))
  for (let r = 0; r < height; r++) {
    side.rows.push(byClass.flatMap((m) => (m[r] !== undefined ? [p.students[m[r]].no, p.students[m[r]].name, tagText(tags[m[r]])] : ['', '', ''])))
    byClass.forEach((m, c) => {
      if (m[r] !== undefined) side.fills!.push(...pairFills(tags[m[r]], r + 1, [c * 3, c * 3 + 1], c * 3 + 2))
    })
  }

  // ペア指定の一覧
  const pairs: SheetData = {
    name: L.sheets.pairs,
    rows: [[L.pairLabel, L.kind, L.members, L.placed, L.judged]],
    fills: [{ row: 0, col: 0, cols: 5, ...HEADER }],
    widths: [6, 8, 40, 16, 8],
  }
  groups.forEach((g, r) => {
    pairs.rows.push([
      g.label,
      g.kind === 'wanted' ? L.wantedKind : L.unwantedKind,
      g.members.map((i) => `${p.students[i].no} ${p.students[i].name}`).join(L.listSep),
      [...new Set(g.members.map((i) => className(classOf[i])))].join(L.joinSep),
      g.ok ? '○' : '×',
    ])
    pairs.fills!.push({ row: r + 1, col: 0, cols: 4, bg: g.color.bg, fg: g.color.fg })
    if (!g.ok) pairs.fills!.push({ row: r + 1, col: 4, bg: VIOLATION_COLOR.bg, fg: VIOLATION_COLOR.fg, bold: true })
  })
  if (groups.length === 0) pairs.rows.push([L.noPairs])

  const summary: SheetData = {
    name: L.sheets.summary,
    rows: [[L.item, L.value, ...byClass.map((_, c) => className(c)), L.ideal]],
    fills: [{ row: 0, col: 0, cols: k + 3, ...HEADER }],
  }
  summary.rows.push([L.count, '', ...report.sizes, round2(p.students.length / k)])
  for (const col of report.columns)
    col.levels.forEach((level, l) =>
      summary.rows.push([col.column, col.kind === 'numeric' ? L.average : level, ...col.rows[l].map(round2), round2(col.ideal[l])]),
    )

  return [
    assign,
    side,
    pairs,
    summary,
    { name: L.sheets.failed, rows: report.violations.length ? report.violations.map((v) => [violationText(lang, v, p, classOf)]) : [[L.none]] },
  ]
}

/** 各組のシート（Excel のみ） */
function classSheet(p: Problem, classOf: number[], c: number, tags: PairTag[][], lang: FileLanguage): SheetData {
  const L = FILE_LABELS[lang]
  const cols = p.columns.map((col) => col.name)
  const sheet: SheetData = {
    name: L.className(c),
    rows: [[L.no, L.name, L.pairCol, ...cols]],
    fills: [{ row: 0, col: 0, cols: 3 + cols.length, ...HEADER }],
    widths: [6, 14, 12, ...cols.map(() => 10)],
  }
  p.students.forEach((s, i) => {
    if (classOf[i] !== c) return
    sheet.rows.push([s.no, s.name, tagText(tags[i]), ...cols.map((col) => toCell(s.values[col]))])
    sheet.fills!.push(...pairFills(tags[i], sheet.rows.length - 1, [0, 1], 2))
  })
  return sheet
}

const rgb = (hex: string) => 'FF' + hex.replace('#', '').toUpperCase()

function toWorksheet(sheet: SheetData) {
  const ws = XLSX.utils.aoa_to_sheet(sheet.rows)
  for (const f of sheet.fills ?? [])
    for (let c = f.col; c < f.col + (f.cols ?? 1); c++) {
      const addr = XLSX.utils.encode_cell({ r: f.row, c })
      ws[addr] ??= { t: 's', v: '' }
      ws[addr].s = {
        ...(f.bg && { fill: { patternType: 'solid', fgColor: { rgb: rgb(f.bg) } } }),
        font: { ...(f.fg && { color: { rgb: rgb(f.fg) } }), ...(f.bold && { bold: true }) },
      }
    }
  if (sheet.widths) ws['!cols'] = sheet.widths.map((wch) => ({ wch }))
  return ws
}

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

/**
 * ブックを .xlsx のバイト列にする。React Native の Blob は ArrayBuffer から作れないため、
 * スマホ版は 'base64' で受け取ってファイルに書き出す。
 */
export function writeXlsx(wb: WorkBook, type: 'base64'): string
export function writeXlsx(wb: WorkBook, type: 'array'): ArrayBuffer
export function writeXlsx(wb: WorkBook, type: 'array' | 'base64'): ArrayBuffer | string {
  return XLSX.write(wb, { bookType: 'xlsx', type })
}

export function exportWorkbook(p: Problem, classOf: number[], k: number, report: Report, lang: FileLanguage = 'ja'): Blob {
  return new Blob([writeXlsx(resultWorkbook(p, classOf, k, report, lang), 'array')], { type: XLSX_MIME })
}

/** 結果のブック（組分け・クラス別名簿・各組・ペア指定・集計・組み合わせ失敗・生徒名簿） */
export function resultWorkbook(p: Problem, classOf: number[], k: number, report: Report, lang: FileLanguage = 'ja'): WorkBook {
  const L = FILE_LABELS[lang]
  const wb = XLSX.utils.book_new()
  const add = (sheet: SheetData) => XLSX.utils.book_append_sheet(wb, toWorksheet(sheet), sheet.name)
  const [assign, side, pairs, summary, failed] = buildResultSheets(p, classOf, k, report, lang)
  const { tags } = pairStatus(p, classOf, L.tagPrefix)

  add(assign)
  add(side)
  for (let c = 0; c < k; c++) add(classSheet(p, classOf, c, tags, lang))
  add(pairs)
  add(summary)
  add(failed)
  add({ name: L.sheets.roster, rows: rosterRows(p, L) })
  add({ name: L.sheets.attributes, rows: attributeRows(p, L) })
  return wb
}

/** 現在の名簿を、ひな形と同じ形式の Excel（再読み込み可能）にする */
export function exportRoster(p: Problem, numClasses: number, lang: FileLanguage = 'ja'): Blob {
  return new Blob([writeXlsx(rosterWorkbook(p, numClasses, lang), 'array')], { type: XLSX_MIME })
}

export function rosterWorkbook(p: Problem, numClasses: number, lang: FileLanguage = 'ja'): WorkBook {
  const L = FILE_LABELS[lang]
  const wb = XLSX.utils.book_new()
  const add = (name: string, rows: unknown[][]) => XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), name)
  const n = p.students.length
  add(L.sheets.settings, [
    [L.studentCount, n],
    [L.maxPerClass, p.maxPerClass !== null && p.maxPerClass * numClasses >= n ? p.maxPerClass : Math.ceil(n / numClasses)],
    [L.classCount, numClasses],
  ])
  add(L.sheets.roster, rosterRows(p, L))
  const nos = (groups: number[][]) => groups.map((g) => g.map((i) => p.students[i].no))
  add(L.sheets.wanted, nos(p.wantedGroups))
  add(L.sheets.unwanted, nos(p.unwantedGroups))
  add(L.sheets.attributes, attributeRows(p, L))
  return wb
}
