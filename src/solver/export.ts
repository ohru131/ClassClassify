import * as XLSX from 'xlsx'
import type { Problem } from './types'
import type { Report } from './evaluate'

export function exportWorkbook(p: Problem, classOf: number[], k: number, report: Report): Blob {
  const wb = XLSX.utils.book_new()
  const add = (name: string, rows: unknown[][]) => XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), name)
  const className = (c: number) => `${c + 1}組`
  const cols = p.columns.map((c) => c.name)

  add('組分け', [
    ['NO', '名前', '組', ...cols],
    ...p.students.map((s, i) => [s.no, s.name, className(classOf[i]), ...cols.map((c) => s.values[c])]),
  ])
  for (let c = 0; c < k; c++) {
    add(className(c), [
      ['NO', '名前', ...cols],
      ...p.students.filter((_, i) => classOf[i] === c).map((s) => [s.no, s.name, ...cols.map((col) => s.values[col])]),
    ])
  }
  const summary: unknown[][] = [['項目', '値', ...Array.from({ length: k }, (_, c) => className(c)), '理想']]
  summary.push(['人数', '', ...report.sizes, p.students.length / k])
  for (const col of report.columns)
    col.levels.forEach((level, l) =>
      summary.push([col.column, level, ...col.rows[l].map((v) => Math.round(v * 100) / 100), Math.round(col.ideal[l] * 100) / 100]),
    )
  add('集計', summary)
  add('組み合わせ失敗', report.violations.length ? report.violations.map((v) => [v.message]) : [['なし']])
  add('生徒名簿', p.rosterSheet)

  const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
  return new Blob([out], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
}
