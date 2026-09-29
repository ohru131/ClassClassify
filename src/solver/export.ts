import * as XLSX from 'xlsx'
import type { Problem } from './types'
import type { Report } from './evaluate'

export type Cell = string | number | null
export interface SheetData {
  name: string
  rows: Cell[][]
}

const className = (c: number) => `${c + 1}組`
const round2 = (v: number) => Math.round(v * 100) / 100

/** 結果の各シート（組分け・クラス別名簿・集計・組み合わせ失敗）の中身を作る */
export function buildResultSheets(p: Problem, classOf: number[], k: number, report: Report): SheetData[] {
  const cols = p.columns.map((c) => c.name)
  const byClass = Array.from({ length: k }, (_, c) => p.students.filter((_, i) => classOf[i] === c))

  const side: Cell[][] = [byClass.flatMap((_, c) => [className(c), ''])]
  const height = Math.max(...byClass.map((m) => m.length))
  for (let r = 0; r < height; r++) side.push(byClass.flatMap((m) => (m[r] ? [m[r].no, m[r].name] : ['', ''])))

  const summary: Cell[][] = [['項目', '値', ...byClass.map((_, c) => className(c)), '理想']]
  summary.push(['人数', '', ...report.sizes, round2(p.students.length / k)])
  for (const col of report.columns)
    col.levels.forEach((level, l) => summary.push([col.column, level, ...col.rows[l].map(round2), round2(col.ideal[l])]))

  return [
    {
      name: '組分け',
      rows: [['NO', '名前', '組', ...cols], ...p.students.map((s, i) => [s.no, s.name, className(classOf[i]), ...cols.map((c) => s.values[c])])],
    },
    { name: 'クラス別名簿', rows: side },
    { name: '集計', rows: summary },
    { name: '組み合わせ失敗', rows: report.violations.length ? report.violations.map((v) => [v.message]) : [['なし']] },
  ]
}

export function exportWorkbook(p: Problem, classOf: number[], k: number, report: Report): Blob {
  const wb = XLSX.utils.book_new()
  const add = (name: string, rows: unknown[][]) => XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), name)
  const cols = p.columns.map((c) => c.name)
  const [assign, side, summary, failed] = buildResultSheets(p, classOf, k, report)

  add(assign.name, assign.rows)
  add(side.name, side.rows)
  for (let c = 0; c < k; c++) {
    add(className(c), [
      ['NO', '名前', ...cols],
      ...p.students.filter((_, i) => classOf[i] === c).map((s) => [s.no, s.name, ...cols.map((col) => s.values[col])]),
    ])
  }
  add(summary.name, summary.rows)
  add(failed.name, failed.rows)
  add('生徒名簿', p.rosterSheet)

  const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
  return new Blob([out], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
}
